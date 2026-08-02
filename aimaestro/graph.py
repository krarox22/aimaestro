"""LangGraph wiring: nodes, routing, and graph construction.

The shape of a turn: the conversational node replies using everything already
remembered, and may call a single ``UpdateMemory`` tool to signal that something
is worth keeping. A conditional edge sends that to one of three writer nodes,
each of which persists to the store and hands control back, so the assistant can
respond with the write already done.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from trustcall import create_extractor

from aimaestro import prompts
from aimaestro.config import Configuration, provider_of
from aimaestro.schemas import Profile, ToDo, UpdateMemory

logger = logging.getLogger(__name__)

PROFILE_NS = "profile"
TODO_NS = "todo"
INSTRUCTIONS_NS = "instructions"
INSTRUCTIONS_KEY = "user_instructions"

_ROUTES = {
    "user": "update_profile",
    "todo": "update_todos",
    "instructions": "update_instructions",
}


# --------------------------------------------------------------------------- #
# Model access
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def get_model(model_id: str):
    """Instantiate a chat model, once per model id.

    Deliberately lazy: building this at import time turns a missing API key into
    an import error instead of an actionable message.
    """
    return init_chat_model(model_id, temperature=0)


def get_profile_extractor(model_id: str):
    """A trustcall extractor that maintains the single Profile record."""
    return create_extractor(get_model(model_id), tools=[Profile], tool_choice="Profile")


def get_todo_extractor(model_id: str, listener=None):
    """A trustcall extractor that inserts and patches ToDo records."""
    extractor = create_extractor(
        get_model(model_id),
        tools=[ToDo],
        tool_choice="ToDo",
        enable_inserts=True,
    )
    if listener is not None:
        extractor = extractor.with_listeners(on_end=listener)
    return extractor


def _bind_memory_tool(model, model_id: str):
    """Bind UpdateMemory, disabling parallel calls only where that is supported.

    ``parallel_tool_calls`` is an OpenAI API parameter; sending it to other
    providers errors at request time.
    """
    if provider_of(model_id) == "openai":
        return model.bind_tools([UpdateMemory], parallel_tool_calls=False)
    return model.bind_tools([UpdateMemory])


# --------------------------------------------------------------------------- #
# Reporting what changed
# --------------------------------------------------------------------------- #


class Spy:
    """Collects trustcall's tool calls so we can describe what changed."""

    def __init__(self) -> None:
        self.called_tools: list = []

    def __call__(self, run) -> None:
        queue = [run]
        while queue:
            current = queue.pop()
            if current.child_runs:
                queue.extend(current.child_runs)
            if current.run_type == "chat_model":
                self.called_tools.append(
                    current.outputs["generations"][0][0]["message"]["kwargs"][
                        "tool_calls"
                    ]
                )


def describe_changes(tool_calls, schema_name: str = "Memory") -> str:
    """Turn trustcall's patches and inserts into a line the model can relay."""
    parts = []
    for group in tool_calls:
        for call in group:
            if call["name"] == "PatchDoc":
                patches = call["args"].get("patches") or [{}]
                parts.append(
                    f"Updated {call['args'].get('json_doc_id')}: "
                    f"{call['args'].get('planned_edits')} "
                    f"({patches[0].get('value')})"
                )
            elif call["name"] == schema_name:
                parts.append(f"New {schema_name}: {call['args']}")
    return "\n\n".join(parts) or f"{schema_name} saved."


def _tool_call_id(state: MessagesState) -> str:
    calls = getattr(state["messages"][-1], "tool_calls", None) or []
    return calls[0]["id"] if calls else "unknown"


def _conversation_for_extraction(state: MessagesState) -> list:
    """The turn so far, minus the tool-call message that triggered this write."""
    instruction = prompts.EXTRACTION_SYSTEM.format(time=datetime.now().isoformat())
    return list(
        merge_message_runs(
            messages=[SystemMessage(content=instruction)] + state["messages"][:-1]
        )
    )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def aimaestro(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reply, using everything remembered about this user."""
    cfg = Configuration.from_runnable_config(config)
    user_id = cfg.user_id

    profile_items = store.search((PROFILE_NS, user_id))
    user_profile = profile_items[0].value if profile_items else None

    todo = "\n".join(str(item.value) for item in store.search((TODO_NS, user_id)))

    instruction_items = store.search((INSTRUCTIONS_NS, user_id))
    instructions = instruction_items[0].value if instruction_items else ""

    system_msg = prompts.ASSISTANT_SYSTEM.format(
        user_profile=user_profile, todo=todo, instructions=instructions
    )
    model = _bind_memory_tool(get_model(cfg.model), cfg.model)
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": [response]}


def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Fold anything newly learned about the user into their profile."""
    cfg = Configuration.from_runnable_config(config)
    namespace = (PROFILE_NS, cfg.user_id)

    existing = store.search(namespace)
    existing_memories = (
        [(item.key, "Profile", item.value) for item in existing] if existing else None
    )

    try:
        result = get_profile_extractor(cfg.model).invoke(
            {
                "messages": _conversation_for_extraction(state),
                "existing": existing_memories,
            }
        )
        for response, meta in zip(result["responses"], result["response_metadata"]):
            store.put(
                namespace,
                meta.get("json_doc_id", str(uuid.uuid4())),
                response.model_dump(mode="json"),
            )
        content = "updated profile"
    except Exception:
        logger.exception("Profile extraction failed")
        content = "could not update the profile this time"

    return {
        "messages": [
            {"role": "tool", "content": content, "tool_call_id": _tool_call_id(state)}
        ]
    }


def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Add to, or amend, the ToDo list."""
    cfg = Configuration.from_runnable_config(config)
    namespace = (TODO_NS, cfg.user_id)

    existing = store.search(namespace)
    existing_memories = (
        [(item.key, "ToDo", item.value) for item in existing] if existing else None
    )

    spy = Spy()
    try:
        result = get_todo_extractor(cfg.model, listener=spy).invoke(
            {
                "messages": _conversation_for_extraction(state),
                "existing": existing_memories,
            }
        )
        for response, meta in zip(result["responses"], result["response_metadata"]):
            store.put(
                namespace,
                meta.get("json_doc_id", str(uuid.uuid4())),
                response.model_dump(mode="json"),
            )
        content = describe_changes(spy.called_tools, "ToDo")
    except Exception:
        logger.exception("ToDo extraction failed")
        content = "could not update the ToDo list this time"

    return {
        "messages": [
            {"role": "tool", "content": content, "tool_call_id": _tool_call_id(state)}
        ]
    }


def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Revise how aiMaestro manages this user's list."""
    cfg = Configuration.from_runnable_config(config)
    namespace = (INSTRUCTIONS_NS, cfg.user_id)

    current = store.get(namespace, INSTRUCTIONS_KEY)
    system_msg = prompts.INSTRUCTIONS_SYSTEM.format(
        current_instructions=current.value if current else None
    )

    try:
        revised = get_model(cfg.model).invoke(
            [SystemMessage(content=system_msg)]
            + state["messages"][:-1]
            + [HumanMessage(content=prompts.INSTRUCTIONS_NUDGE)]
        )
        store.put(namespace, INSTRUCTIONS_KEY, {"memory": revised.content})
        content = "updated instructions"
    except Exception:
        logger.exception("Instruction update failed")
        content = "could not update instructions this time"

    return {
        "messages": [
            {"role": "tool", "content": content, "tool_call_id": _tool_call_id(state)}
        ]
    }


# --------------------------------------------------------------------------- #
# Routing and construction
# --------------------------------------------------------------------------- #


def route_message(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
    """Send the turn to a memory writer, or end it."""
    message = state["messages"][-1]
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return END

    update_type = tool_calls[0].get("args", {}).get("update_type")
    target = _ROUTES.get(update_type)
    if target is None:
        # A malformed argument should not take down the conversation.
        logger.warning("Unrecognized update_type %r; ending turn.", update_type)
        return END
    return target


def make_builder() -> StateGraph:
    """Assemble the graph topology."""
    builder = StateGraph(MessagesState, context_schema=Configuration)
    builder.add_node("aimaestro", aimaestro)
    builder.add_node("update_profile", update_profile)
    builder.add_node("update_todos", update_todos)
    builder.add_node("update_instructions", update_instructions)

    builder.add_edge(START, "aimaestro")
    builder.add_conditional_edges("aimaestro", route_message)
    builder.add_edge("update_profile", "aimaestro")
    builder.add_edge("update_todos", "aimaestro")
    builder.add_edge("update_instructions", "aimaestro")
    return builder


def build_graph(backend=None):
    """Compile the graph.

    With a backend, memory and threads persist to SQLite — this is what the CLI
    uses. Without one, the caller (LangGraph Studio) supplies its own.
    """
    builder = make_builder()
    if backend is None:
        return builder.compile()
    return builder.compile(checkpointer=backend.checkpointer, store=backend.store)


#: Entry point for LangGraph Studio, which injects its own store and checkpointer.
graph = make_builder().compile()

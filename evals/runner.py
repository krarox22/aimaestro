"""Replay a scenario against a model and capture everything worth scoring."""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from aimaestro.graph import (
    INSTRUCTIONS_KEY,
    INSTRUCTIONS_NS,
    PROFILE_NS,
    TODO_NS,
    _ROUTES,
    build_graph,
)
from aimaestro.store import MemoryBackend
from evals.scenarios import Scenario

EVAL_USER = "eval-user"

#: Seeded records get predictable keys so the integrity scorer can compare a
#: final record against exactly the record it started as.
PROFILE_SEED_KEY = "seed-profile"


class _PromptRecorder(BaseCallbackHandler):
    """Captures the system prompt handed to the model on every call.

    A callback rather than a wrapper, so this behaves identically whether the
    model underneath is the test fake or a real provider.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        for batch in messages:
            if batch:
                self.prompts.append(str(batch[0].content))


@dataclass
class RunResult:
    """Everything one replay of a scenario produced."""

    scenario_id: str
    routes: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    seed_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


def _seed(store, seed: dict) -> None:
    profile = seed.get("profile")
    if profile:
        store.put((PROFILE_NS, EVAL_USER), PROFILE_SEED_KEY, dict(profile))

    for index, todo in enumerate(seed.get("todo") or []):
        store.put((TODO_NS, EVAL_USER), f"seed-todo-{index}", dict(todo))

    instructions = seed.get("instructions")
    if instructions:
        store.put(
            (INSTRUCTIONS_NS, EVAL_USER),
            INSTRUCTIONS_KEY,
            {"memory": instructions},
        )


def _snapshot(store) -> dict[str, dict[str, Any]]:
    return {
        name: {item.key: item.value for item in store.search((name, EVAL_USER))}
        for name in (PROFILE_NS, TODO_NS, INSTRUCTIONS_NS)
    }


def _routes_from(messages) -> list[str]:
    """Recover the memory writers a turn visited, from its message history."""
    routes = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("name") != "UpdateMemory":
                continue
            target = _ROUTES.get(call.get("args", {}).get("update_type"))
            if target:
                routes.append(target)
    return routes


def run_scenario(
    scenario: Scenario, model_id: str, db_path: str | None = None
) -> RunResult:
    """Seed memory, replay every turn, and capture the outcome.

    Each run gets its own database unless one is supplied, so repeated runs of
    the same scenario cannot contaminate each other.
    """
    result = RunResult(scenario_id=scenario.id)
    temp_dir = None
    if db_path is None:
        temp_dir = tempfile.mkdtemp(prefix="aimaestro-eval-")
        db_path = str(Path(temp_dir) / "eval.db")

    recorder = _PromptRecorder()
    thread_id = str(uuid.uuid4())

    try:
        with MemoryBackend(db_path) as backend:
            _seed(backend.store, scenario.seed)
            result.seed_memory = _snapshot(backend.store)

            app = build_graph(backend)
            config = {
                "configurable": {"user_id": EVAL_USER, "thread_id": thread_id},
                "callbacks": [recorder],
            }

            for turn in scenario.turns:
                output = app.invoke({"messages": [("user", turn)]}, config)
                result.routes.extend(_routes_from(output["messages"]))
                result.replies.append(str(output["messages"][-1].content))

            result.final_memory = _snapshot(backend.store)
    except Exception as exc:  # a broken run is data, not a crash
        result.error = f"{type(exc).__name__}: {exc}"

    result.prompts = recorder.prompts
    return result

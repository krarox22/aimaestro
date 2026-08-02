# aiMaestro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a long-term-memory task assistant that remembers your profile, ToDo list, and preferences across restarts, runnable from the terminal and from LangGraph Studio.

**Architecture:** A LangGraph graph with one conversational node bound to a single `UpdateMemory` tool; the model calling that tool routes to one of three writer nodes (profile / todos / instructions), each persisting through trustcall JSON-patch extraction into a `SqliteStore`. Conversation threads persist via `SqliteSaver` in the same database file.

**Tech Stack:** Python 3.12, langgraph 1.2, langgraph-checkpoint-sqlite 3.1, langchain-core 1.5, trustcall 0.0.39, pydantic v2, pytest, `uv` for env management.

**Working directory:** `/Users/kratikahimanshu/ai mistro` (note the space — always quote paths in shell commands).

---

## Verified environment facts

These were confirmed empirically before writing this plan. Do not re-litigate them:

- `langgraph.store.sqlite.SqliteStore` and `langgraph.checkpoint.sqlite.SqliteSaver` both exist.
- **`SqliteStore` requires `isolation_level=None`** on its connection. Without it, every `put()` raises `sqlite3.OperationalError: cannot start a transaction within a transaction`.
- Store and saver can share one database file using separate connections.
- Memory written, connection closed, and store reopened from the same file returns the data intact.
- `parallel_tool_calls=False` is accepted by Gemini at `bind_tools()` time but is an OpenAI-only API parameter, so it must be applied conditionally.

---

## File structure

| File | Responsibility |
|---|---|
| `aimaestro/config.py` | Env/config resolution, provider→API-key mapping, missing-key detection. |
| `aimaestro/schemas.py` | `Profile`, `ToDo`, `UpdateMemory`. Pure data, no logic. |
| `aimaestro/prompts.py` | All prompt strings. No imports from the rest of the package. |
| `aimaestro/store.py` | Opening/closing SQLite store + saver. Owns connection settings. |
| `aimaestro/graph.py` | Nodes, routing, graph construction. |
| `aimaestro/cli.py` | Terminal chat loop, slash commands. |
| `aimaestro/__main__.py` | `python -m aimaestro` entry point. |
| `tests/conftest.py` | `FakeToolCallingModel` and shared fixtures. |
| `tests/test_config.py` | Config precedence and key detection. |
| `tests/test_store.py` | Persistence, including the restart test. |
| `tests/test_routing.py` | Conditional-edge behavior. |
| `tests/test_graph_flow.py` | End-to-end graph with fake model and stub extractors. |

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.env.example`, `LICENSE`, `aimaestro/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create the virtualenv**

```bash
cd "/Users/kratikahimanshu/ai mistro"
uv venv --python 3.12 .venv
```

Expected: `Creating virtual environment at: .venv`

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "aimaestro"
version = "0.1.0"
description = "A task assistant with long-term memory."
requires-python = ">=3.12,<3.13"
dependencies = [
    "langgraph>=1.2,<2",
    "langgraph-checkpoint-sqlite>=3.1,<4",
    "langchain>=1.3,<2",
    "langchain-core>=1.5,<2",
    "trustcall>=0.0.39",
    "langchain-google-genai>=4.3,<5",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
openai = ["langchain-openai>=1.0"]
anthropic = ["langchain-anthropic>=1.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[project.scripts]
aimaestro = "aimaestro.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 3: Install dependencies**

```bash
cd "/Users/kratikahimanshu/ai mistro"
uv pip install --python .venv/bin/python -e ".[dev]"
```

Expected: exit code 0.

- [ ] **Step 4: Write `.env.example`**

```bash
# Which model powers aiMaestro. Format: provider:model
AIMAESTRO_MODEL=google_genai:gemini-2.5-flash

# Set the key matching your provider above.
# Free Gemini key: https://aistudio.google.com/apikey
GOOGLE_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=

# Optional: separate memory profiles
AIMAESTRO_USER_ID=default-user
```

- [ ] **Step 5: Write `LICENSE`**

MIT text retaining the upstream copyright line, with the user's line added:

```
MIT License

Copyright (c) 2026 LangChain, Inc.
Copyright (c) 2026 Kratika Agarwal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 6: Create empty package files**

```bash
cd "/Users/kratikahimanshu/ai mistro"
mkdir -p aimaestro tests data
touch aimaestro/__init__.py tests/__init__.py
```

- [ ] **Step 7: Commit**

```bash
cd "/Users/kratikahimanshu/ai mistro"
git add -A && git commit -m "chore: project scaffolding and dependencies"
```

---

### Task 2: Configuration

**Files:**
- Create: `aimaestro/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from aimaestro.config import Configuration, api_key_error, DEFAULT_MODEL


def test_defaults_when_no_env(monkeypatch):
    monkeypatch.delenv("AIMAESTRO_MODEL", raising=False)
    monkeypatch.delenv("AIMAESTRO_USER_ID", raising=False)
    cfg = Configuration.from_runnable_config(None)
    assert cfg.model == DEFAULT_MODEL
    assert cfg.user_id == "default-user"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("AIMAESTRO_MODEL", "openai:gpt-4o")
    cfg = Configuration.from_runnable_config(None)
    assert cfg.model == "openai:gpt-4o"


def test_env_var_beats_runnable_config(monkeypatch):
    monkeypatch.setenv("AIMAESTRO_USER_ID", "from-env")
    cfg = Configuration.from_runnable_config({"configurable": {"user_id": "from-config"}})
    assert cfg.user_id == "from-env"


def test_runnable_config_used_when_env_absent(monkeypatch):
    monkeypatch.delenv("AIMAESTRO_USER_ID", raising=False)
    cfg = Configuration.from_runnable_config({"configurable": {"user_id": "kratika"}})
    assert cfg.user_id == "kratika"


def test_api_key_error_when_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    err = api_key_error("google_genai:gemini-2.5-flash")
    assert err is not None and "GOOGLE_API_KEY" in err


def test_api_key_error_detects_placeholder(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "your-google-api-key-here")
    err = api_key_error("google_genai:gemini-2.5-flash")
    assert err is not None and "placeholder" in err.lower()


def test_api_key_error_none_when_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyRealLookingKey123")
    assert api_key_error("google_genai:gemini-2.5-flash") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aimaestro.config'`

- [ ] **Step 3: Write `aimaestro/config.py`**

```python
"""Environment-driven configuration for aiMaestro."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

DEFAULT_MODEL = "google_genai:gemini-2.5-flash"
DEFAULT_DB_PATH = "data/aimaestro.db"
ENV_PREFIX = "AIMAESTRO_"

#: Which API key each provider needs.
PROVIDER_KEYS = {
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

#: Values that look like a key but are not one.
_PLACEHOLDER_MARKERS = ("your-", "sk-xxx", "changeme", "<", "xxx")


@dataclass(kw_only=True)
class Configuration:
    """Settings resolved from the environment, then the runnable config."""

    user_id: str = "default-user"
    model: str = DEFAULT_MODEL
    db_path: str = DEFAULT_DB_PATH

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {}
        for f in fields(cls):
            if not f.init:
                continue
            values[f.name] = os.environ.get(
                f"{ENV_PREFIX}{f.name.upper()}"
            ) or configurable.get(f.name)
        return cls(**{k: v for k, v in values.items() if v})


def provider_of(model: str) -> str | None:
    """The provider half of a ``provider:model`` identifier."""
    return model.split(":", 1)[0] if ":" in model else None


def api_key_error(model: str) -> str | None:
    """Return a human-readable problem with the API key, or None if it looks usable."""
    var = PROVIDER_KEYS.get(provider_of(model) or "")
    if var is None:
        return None

    value = os.environ.get(var, "").strip()
    if not value:
        return (
            f"{var} is not set, which {model} needs.\n"
            f"Add it to your .env file. A free Gemini key: "
            f"https://aistudio.google.com/apikey"
        )
    if any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS):
        return (
            f"{var} is still set to a placeholder value ({value!r}).\n"
            f"Replace it with a real key in your .env file."
        )
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_config.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add aimaestro/config.py tests/test_config.py && git commit -m "feat: env-driven configuration with API key validation"
```

---

### Task 3: Schemas

**Files:**
- Create: `aimaestro/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

```python
from aimaestro.schemas import Profile, ToDo, UpdateMemory


def test_profile_all_fields_optional():
    p = Profile()
    assert p.name is None
    assert p.connections == []
    assert p.interests == []


def test_todo_requires_only_task():
    t = ToDo(task="renew passport")
    assert t.task == "renew passport"
    assert t.status == "not started"
    assert t.solutions == []
    assert t.deadline is None


def test_todo_solutions_accepts_empty_list():
    """Regression: the original schema declared min_items=1 with an empty
    default, which are contradictory."""
    t = ToDo(task="x", solutions=[])
    assert t.solutions == []


def test_update_memory_literal_values():
    assert set(UpdateMemory.__annotations__["update_type"].__args__) == {
        "user",
        "todo",
        "instructions",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aimaestro.schemas'`

- [ ] **Step 3: Write `aimaestro/schemas.py`**

```python
"""Data shapes aiMaestro remembers, and the tool it uses to decide to remember."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Profile(BaseModel):
    """What aiMaestro knows about the person it is talking to."""

    name: Optional[str] = Field(default=None, description="The user's name")
    location: Optional[str] = Field(default=None, description="Where the user lives")
    job: Optional[str] = Field(default=None, description="What the user does for work")
    connections: list[str] = Field(
        default_factory=list,
        description="People in the user's life: family, friends, coworkers",
    )
    interests: list[str] = Field(
        default_factory=list, description="Things the user is interested in"
    )


class ToDo(BaseModel):
    """A single task on the user's list."""

    task: str = Field(description="The task to be completed")
    time_to_complete: Optional[int] = Field(
        default=None, description="Estimated minutes to complete"
    )
    deadline: Optional[datetime] = Field(
        default=None, description="When this needs to be done, if there is a date"
    )
    solutions: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete, actionable next steps: specific services, contacts, or "
            "options relevant to finishing the task"
        ),
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        default="not started", description="Current status of the task"
    )


class UpdateMemory(TypedDict):
    """Signal that something from this conversation is worth remembering."""

    update_type: Literal["user", "todo", "instructions"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_schemas.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add aimaestro/schemas.py tests/test_schemas.py && git commit -m "feat: memory schemas"
```

---

### Task 4: Prompts

**Files:**
- Create: `aimaestro/prompts.py`

No tests: this module is pure string constants, exercised by Task 7's flow tests.

- [ ] **Step 1: Write `aimaestro/prompts.py`**

```python
"""Prompt text for aiMaestro. Kept apart from logic so it can be tuned freely."""

ASSISTANT_SYSTEM = """You are aiMaestro, a task assistant with a long memory.

You help one person stay on top of what they need to do. You remember three
things about them between conversations:

1. Their profile — who they are and what matters to them.
2. Their ToDo list.
3. Instructions — how they have told you they like their list managed.

Their profile so far (may be empty):
<user_profile>
{user_profile}
</user_profile>

Their current ToDo list (may be empty):
<todo>
{todo}
</todo>

Preferences they have given you about managing the list (may be empty):
<instructions>
{instructions}
</instructions>

How to handle each message:

1. Read what they said carefully, in the context of what you already know.

2. Decide whether anything should be committed to long-term memory, and if so
   call the UpdateMemory tool:
   - They revealed something personal — call it with update_type "user"
   - They mentioned a task, or a change to one — call it with update_type "todo"
   - They told you how they want the list handled — update_type "instructions"

3. Be selective about what you say out loud:
   - Never announce that you updated their profile.
   - Do tell them when you change their ToDo list.
   - Never announce that you updated your instructions.

4. Lean toward capturing tasks. Do not ask permission before saving one.

5. After saving, reply naturally. If nothing needed saving, just reply.
"""

EXTRACTION_SYSTEM = """Review the conversation below.

Use the supplied tools to record anything worth remembering about this person.

Where several records need creating or amending, handle them in parallel.

Current time: {time}"""

INSTRUCTIONS_SYSTEM = """Review the conversation below.

Update your standing instructions for managing this person's ToDo list, based on
any preference they expressed — how they like items worded, what detail they
want, what they find useful.

Your current instructions:

<current_instructions>
{current_instructions}
</current_instructions>"""

INSTRUCTIONS_NUDGE = "Please update the instructions based on our conversation."
```

- [ ] **Step 2: Commit**

```bash
git add aimaestro/prompts.py && git commit -m "feat: prompt text"
```

---

### Task 5: SQLite persistence layer

**Files:**
- Create: `aimaestro/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from aimaestro.store import MemoryBackend


def test_put_and_search(tmp_path):
    db = str(tmp_path / "t.db")
    with MemoryBackend(db) as backend:
        backend.store.put(("profile", "kratika"), "p1", {"name": "Kratika"})
        found = backend.store.search(("profile", "kratika"))
    assert [i.value for i in found] == [{"name": "Kratika"}]


def test_memory_survives_restart(tmp_path):
    """The central guarantee: memory outlives the process that wrote it."""
    db = str(tmp_path / "t.db")

    with MemoryBackend(db) as backend:
        backend.store.put(("todo", "kratika"), "t1", {"task": "renew passport"})

    # Entirely new backend, new connections, same file.
    with MemoryBackend(db) as backend:
        found = backend.store.search(("todo", "kratika"))

    assert [i.value for i in found] == [{"task": "renew passport"}]


def test_checkpointer_and_store_share_a_file(tmp_path):
    db = str(tmp_path / "t.db")
    with MemoryBackend(db) as backend:
        assert backend.store is not None
        assert backend.checkpointer is not None
        backend.store.put(("profile", "u"), "k", {"a": 1})
        assert backend.store.get(("profile", "u"), "k").value == {"a": 1}


def test_creates_parent_directory(tmp_path):
    db = str(tmp_path / "nested" / "dir" / "t.db")
    with MemoryBackend(db) as backend:
        backend.store.put(("profile", "u"), "k", {"a": 1})
    assert (tmp_path / "nested" / "dir" / "t.db").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aimaestro.store'`

- [ ] **Step 3: Write `aimaestro/store.py`**

Note the `isolation_level=None`: `SqliteStore` issues its own `BEGIN`, so the
connection must be in autocommit mode or every write raises
`cannot start a transaction within a transaction`.

```python
"""SQLite-backed persistence: long-term memory plus conversation threads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


def _connect(db_path: str) -> sqlite3.Connection:
    """Open a connection configured the way langgraph's SQLite backends expect."""
    return sqlite3.connect(
        db_path,
        check_same_thread=False,
        # Autocommit. SqliteStore manages its own transactions and will fail
        # with "cannot start a transaction within a transaction" otherwise.
        isolation_level=None,
    )


class MemoryBackend:
    """Owns the database connections for one aiMaestro session.

    Use as a context manager so connections are always closed::

        with MemoryBackend("data/aimaestro.db") as backend:
            backend.store.put(...)
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Separate connections: the store and the checkpointer each manage
        # their own cursors and transaction state.
        self._store_conn = _connect(db_path)
        self._saver_conn = _connect(db_path)

        self.store = SqliteStore(self._store_conn)
        self.store.setup()

        self.checkpointer = SqliteSaver(self._saver_conn)
        self.checkpointer.setup()

    def close(self) -> None:
        self._store_conn.close()
        self._saver_conn.close()

    def __enter__(self) -> "MemoryBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_store.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add aimaestro/store.py tests/test_store.py && git commit -m "feat: sqlite persistence surviving restarts"
```

---

### Task 6: Graph routing

**Files:**
- Create: `aimaestro/graph.py` (routing portion only)
- Test: `tests/test_routing.py`

- [ ] **Step 1: Write the failing tests**

```python
from langchain_core.messages import AIMessage
from langgraph.graph import END

from aimaestro.graph import route_message


def _ai_with_update(update_type):
    return AIMessage(
        content="",
        tool_calls=[{"name": "UpdateMemory", "args": {"update_type": update_type}, "id": "1"}],
    )


def test_no_tool_calls_ends():
    state = {"messages": [AIMessage(content="hello", tool_calls=[])]}
    assert route_message(state, None, None) == END


def test_user_routes_to_profile():
    state = {"messages": [_ai_with_update("user")]}
    assert route_message(state, None, None) == "update_profile"


def test_todo_routes_to_todos():
    state = {"messages": [_ai_with_update("todo")]}
    assert route_message(state, None, None) == "update_todos"


def test_instructions_routes_to_instructions():
    state = {"messages": [_ai_with_update("instructions")]}
    assert route_message(state, None, None) == "update_instructions"


def test_unknown_update_type_ends_instead_of_raising():
    """Regression: the original raised ValueError, killing the chat session on a
    malformed tool argument."""
    state = {"messages": [_ai_with_update("nonsense")]}
    assert route_message(state, None, None) == END
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_routing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aimaestro.graph'`

- [ ] **Step 3: Write the routing portion of `aimaestro/graph.py`**

```python
"""LangGraph wiring: nodes, routing, and graph construction."""

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_routing.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add aimaestro/graph.py tests/test_routing.py && git commit -m "feat: memory routing that degrades safely"
```

---

### Task 7: Graph nodes and construction

**Files:**
- Modify: `aimaestro/graph.py` (append nodes + builder)
- Create: `tests/conftest.py`, `tests/test_graph_flow.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Test doubles that let the whole graph run without an API key."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolCallingModel(BaseChatModel):
    """Replays a scripted list of AIMessages, one per invocation.

    The last message repeats once the script runs out, so a graph that loops
    back to the conversational node terminates instead of hanging.
    """

    responses: list[AIMessage] = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        index = min(self.calls, len(self.responses) - 1)
        message = self.responses[index]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self


class StubExtractor:
    """Stands in for a trustcall extractor with a canned result."""

    def __init__(self, responses: list[Any], metadata: list[dict] | None = None):
        self._responses = responses
        self._metadata = metadata or [{} for _ in responses]
        self.invocations: list[dict] = []

    def invoke(self, payload: dict) -> dict:
        self.invocations.append(payload)
        return {"responses": self._responses, "response_metadata": self._metadata}

    def with_listeners(self, **kwargs: Any) -> "StubExtractor":
        return self


def update_memory_call(update_type: str, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "UpdateMemory", "args": {"update_type": update_type}, "id": call_id}
        ],
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "aimaestro.db")
```

- [ ] **Step 2: Write the failing flow tests**

```python
import pytest
from langchain_core.messages import AIMessage

from aimaestro import graph as graph_module
from aimaestro.graph import (
    INSTRUCTIONS_KEY,
    INSTRUCTIONS_NS,
    PROFILE_NS,
    TODO_NS,
    build_graph,
)
from aimaestro.schemas import Profile, ToDo
from aimaestro.store import MemoryBackend
from tests.conftest import FakeToolCallingModel, StubExtractor, update_memory_call

CONFIG = {"configurable": {"user_id": "kratika", "thread_id": "t1"}}


@pytest.fixture
def fake_model(monkeypatch):
    def install(responses):
        model = FakeToolCallingModel(responses=responses)
        monkeypatch.setattr(graph_module, "get_model", lambda model_id: model)
        return model

    return install


def test_plain_reply_writes_nothing(db_path, fake_model):
    fake_model([AIMessage(content="Hi, what can I help with?")])
    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        result = app.invoke({"messages": [("user", "hello")]}, CONFIG)
        assert backend.store.search((TODO_NS, "kratika")) == []
    assert "help" in result["messages"][-1].content


def test_profile_update_is_persisted(db_path, fake_model, monkeypatch):
    fake_model([update_memory_call("user"), AIMessage(content="Good to know.")])
    monkeypatch.setattr(
        graph_module,
        "get_profile_extractor",
        lambda model_id: StubExtractor([Profile(name="Kratika", interests=["langgraph"])]),
    )
    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        app.invoke({"messages": [("user", "I'm Kratika and I like langgraph")]}, CONFIG)
        saved = [i.value for i in backend.store.search((PROFILE_NS, "kratika"))]
    assert saved[0]["name"] == "Kratika"


def test_todo_update_is_persisted(db_path, fake_model, monkeypatch):
    fake_model([update_memory_call("todo"), AIMessage(content="Added it.")])
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="renew passport")]),
    )
    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        app.invoke({"messages": [("user", "I need to renew my passport")]}, CONFIG)
        saved = [i.value for i in backend.store.search((TODO_NS, "kratika"))]
    assert saved[0]["task"] == "renew passport"


def test_instructions_update_is_persisted(db_path, fake_model):
    fake_model(
        [
            update_memory_call("instructions"),
            AIMessage(content="Always include a deadline."),
            AIMessage(content="Understood."),
        ]
    )
    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        app.invoke({"messages": [("user", "always add deadlines")]}, CONFIG)
        saved = backend.store.get((INSTRUCTIONS_NS, "kratika"), INSTRUCTIONS_KEY)
    assert saved is not None and "deadline" in saved.value["memory"].lower()


def test_memory_is_visible_to_a_later_session(db_path, fake_model, monkeypatch):
    """Write a todo in one graph, then read it back in a freshly built one."""
    fake_model([update_memory_call("todo"), AIMessage(content="Added.")])
    monkeypatch.setattr(
        graph_module,
        "get_todo_extractor",
        lambda model_id, listener=None: StubExtractor([ToDo(task="book vet")]),
    )
    with MemoryBackend(db_path) as backend:
        build_graph(backend).invoke({"messages": [("user", "book the vet")]}, CONFIG)

    with MemoryBackend(db_path) as backend:
        saved = [i.value for i in backend.store.search((TODO_NS, "kratika"))]
    assert saved[0]["task"] == "book vet"


def test_extraction_failure_does_not_crash_the_chat(db_path, fake_model, monkeypatch):
    fake_model([update_memory_call("todo"), AIMessage(content="Sorry, trouble saving.")])

    class Exploding:
        def invoke(self, payload):
            raise RuntimeError("model refused")

        def with_listeners(self, **kwargs):
            return self

    monkeypatch.setattr(
        graph_module, "get_todo_extractor", lambda model_id, listener=None: Exploding()
    )
    with MemoryBackend(db_path) as backend:
        result = build_graph(backend).invoke(
            {"messages": [("user", "add a task")]}, CONFIG
        )
    assert result["messages"][-1].content  # conversation continued
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_graph_flow.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_graph'`

- [ ] **Step 4: Append nodes and construction to `aimaestro/graph.py`**

```python
@lru_cache(maxsize=None)
def get_model(model_id: str):
    """Instantiate a chat model. Cached so each model is built once.

    Deliberately lazy: building at import time makes a missing API key surface
    as an import error rather than an actionable message.
    """
    return init_chat_model(model_id, temperature=0)


def get_profile_extractor(model_id: str):
    return create_extractor(
        get_model(model_id), tools=[Profile], tool_choice="Profile"
    )


def get_todo_extractor(model_id: str, listener=None):
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

    ``parallel_tool_calls`` is an OpenAI API parameter; passing it to other
    providers errors at request time.
    """
    if provider_of(model_id) == "openai":
        return model.bind_tools([UpdateMemory], parallel_tool_calls=False)
    return model.bind_tools([UpdateMemory])


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

    instruction = prompts.EXTRACTION_SYSTEM.format(time=datetime.now().isoformat())
    messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=instruction)] + state["messages"][:-1]
        )
    )

    try:
        result = get_profile_extractor(cfg.model).invoke(
            {"messages": messages, "existing": existing_memories}
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
    """Add to or amend the ToDo list."""
    cfg = Configuration.from_runnable_config(config)
    namespace = (TODO_NS, cfg.user_id)

    existing = store.search(namespace)
    existing_memories = (
        [(item.key, "ToDo", item.value) for item in existing] if existing else None
    )

    instruction = prompts.EXTRACTION_SYSTEM.format(time=datetime.now().isoformat())
    messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=instruction)] + state["messages"][:-1]
        )
    )

    spy = Spy()
    try:
        result = get_todo_extractor(cfg.model, listener=spy).invoke(
            {"messages": messages, "existing": existing_memories}
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
    return builder.compile(
        checkpointer=backend.checkpointer, store=backend.store
    )


#: Entry point for LangGraph Studio, which injects its own store/checkpointer.
graph = make_builder().compile()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_graph_flow.py -v`
Expected: 6 passed

- [ ] **Step 6: Run the whole suite**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest -v`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add aimaestro/graph.py tests/conftest.py tests/test_graph_flow.py && git commit -m "feat: memory nodes and graph construction"
```

---

### Task 8: CLI

**Files:**
- Create: `aimaestro/cli.py`, `aimaestro/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
from aimaestro.cli import format_memory
from aimaestro.store import MemoryBackend


def test_format_memory_empty(db_path):
    with MemoryBackend(db_path) as backend:
        text = format_memory(backend.store, "kratika")
    assert "nothing" in text.lower()


def test_format_memory_shows_all_three(db_path):
    with MemoryBackend(db_path) as backend:
        backend.store.put(("profile", "kratika"), "p", {"name": "Kratika"})
        backend.store.put(("todo", "kratika"), "t", {"task": "renew passport"})
        backend.store.put(
            ("instructions", "kratika"), "user_instructions", {"memory": "add deadlines"}
        )
        text = format_memory(backend.store, "kratika")
    assert "Kratika" in text
    assert "renew passport" in text
    assert "add deadlines" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aimaestro.cli'`

- [ ] **Step 3: Write `aimaestro/cli.py`**

```python
"""Terminal chat interface for aiMaestro."""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv

from aimaestro.config import Configuration, api_key_error
from aimaestro.graph import (
    INSTRUCTIONS_KEY,
    INSTRUCTIONS_NS,
    PROFILE_NS,
    TODO_NS,
    build_graph,
)
from aimaestro.store import MemoryBackend

BANNER = """aiMaestro — a task assistant that remembers.

  /memory   show what I know about you
  /reset    start a fresh conversation (long-term memory is kept)
  /quit     exit
"""


def format_memory(store, user_id: str) -> str:
    """Render everything remembered about a user."""
    sections: list[str] = []

    profile = store.search((PROFILE_NS, user_id))
    if profile:
        lines = [
            f"  {key}: {value}"
            for key, value in profile[0].value.items()
            if value not in (None, [], "")
        ]
        if lines:
            sections.append("Profile:\n" + "\n".join(lines))

    todos = store.search((TODO_NS, user_id))
    if todos:
        lines = []
        for item in todos:
            value = item.value
            line = f"  [{value.get('status', '?')}] {value.get('task', '')}"
            if value.get("deadline"):
                line += f" (by {value['deadline']})"
            for solution in value.get("solutions") or []:
                line += f"\n      - {solution}"
            lines.append(line)
        sections.append("ToDo:\n" + "\n".join(lines))

    instructions = store.get((INSTRUCTIONS_NS, user_id), INSTRUCTIONS_KEY)
    if instructions and instructions.value.get("memory"):
        sections.append("Instructions:\n  " + instructions.value["memory"])

    if not sections:
        return "I don't know anything about you yet — nothing saved so far."
    return "\n\n".join(sections)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="aimaestro", description=__doc__)
    parser.add_argument("--user-id", help="Which memory profile to use")
    parser.add_argument("--thread", help="Conversation thread id to resume")
    parser.add_argument("--db", help="Path to the database file")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv()
    args = _parse_args(argv)

    cfg = Configuration.from_runnable_config(
        {"configurable": {k: v for k, v in vars(args).items() if v}}
    )
    user_id = args.user_id or cfg.user_id
    db_path = args.db or cfg.db_path

    problem = api_key_error(cfg.model)
    if problem:
        print(f"Cannot start: {problem}", file=sys.stderr)
        return 1

    thread_id = args.thread or str(uuid.uuid4())
    print(BANNER)
    print(f"model: {cfg.model}   profile: {user_id}\n")

    with MemoryBackend(db_path) as backend:
        app = build_graph(backend)
        while True:
            try:
                user_input = input("you › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not user_input:
                continue
            if user_input in ("/quit", "/exit"):
                return 0
            if user_input == "/memory":
                print("\n" + format_memory(backend.store, user_id) + "\n")
                continue
            if user_input == "/reset":
                thread_id = str(uuid.uuid4())
                print("\nStarted a new conversation. I still remember you.\n")
                continue

            config = {
                "configurable": {"user_id": user_id, "thread_id": thread_id}
            }
            try:
                result = app.invoke({"messages": [("user", user_input)]}, config)
            except Exception as exc:
                print(f"\n[error] {exc}\n", file=sys.stderr)
                continue
            print(f"\naiMaestro › {result['messages'][-1].content}\n")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write `aimaestro/__main__.py`**

```python
from aimaestro.cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 6: Verify the missing-key path gives a clean message**

```bash
cd "/Users/kratikahimanshu/ai mistro"
env -u GOOGLE_API_KEY .venv/bin/python -m aimaestro
```

Expected: `Cannot start: GOOGLE_API_KEY is not set...` and exit code 1 — not a traceback.

- [ ] **Step 7: Commit**

```bash
git add aimaestro/cli.py aimaestro/__main__.py tests/test_cli.py && git commit -m "feat: terminal chat interface"
```

---

### Task 9: Studio registration and README

**Files:**
- Create: `langgraph.json`, `README.md`

- [ ] **Step 1: Write `langgraph.json`**

```json
{
  "dockerfile_lines": [],
  "graphs": {
    "aimaestro": "./aimaestro/graph.py:graph"
  },
  "env": "./.env",
  "python_version": "3.12",
  "dependencies": ["."]
}
```

- [ ] **Step 2: Write `README.md`**

Contains: what it does, quickstart (clone → `uv venv` → install → copy `.env.example` → add key → `python -m aimaestro`), an example session transcript, the slash commands, how memory is organized (the namespace table), how to switch model providers, how to run tests, and how to open it in LangGraph Studio via `langgraph dev`. No references to any course.

- [ ] **Step 3: Verify Studio import works**

```bash
cd "/Users/kratikahimanshu/ai mistro"
GOOGLE_API_KEY=placeholder-not-used .venv/bin/python -c "from aimaestro.graph import graph; print('studio graph ok:', graph)"
```

Expected: `studio graph ok: <langgraph.graph.state.CompiledStateGraph object ...>`

- [ ] **Step 4: Commit**

```bash
git add langgraph.json README.md && git commit -m "docs: readme and studio registration"
```

---

### Task 10: Full verification and publish

- [ ] **Step 1: Run the entire suite**

Run: `cd "/Users/kratikahimanshu/ai mistro" && .venv/bin/pytest -v`
Expected: all tests pass, zero network calls, no API key needed.

- [ ] **Step 2: Confirm nothing secret is tracked**

```bash
cd "/Users/kratikahimanshu/ai mistro"
git status --short
git ls-files | grep -E "\.env$|\.db$" && echo "PROBLEM: secret or db tracked" || echo "clean"
```

Expected: `clean`

- [ ] **Step 3: Create the GitHub repo and push**

```bash
cd "/Users/kratikahimanshu/ai mistro"
git branch -M main
gh repo create aimaestro --public --source=. --remote=origin \
  --description "A task assistant with long-term memory: it learns your profile, tracks your ToDo list, and adapts to how you like it managed." \
  --push
```

Expected: repo URL printed.

- [ ] **Step 4: Confirm the push**

```bash
cd "/Users/kratikahimanshu/ai mistro" && git log --oneline -1 && gh repo view --json url --jq .url
```

---

## Post-implementation: the live check

This requires the user's own API key and is **not** part of the automated suite.

```bash
cd "/Users/kratikahimanshu/ai mistro"
cp .env.example .env       # then paste a real key from https://aistudio.google.com/apikey
.venv/bin/python -m aimaestro
```

Suggested script: say `I'm Kratika, I live in Atlanta and I need to renew my passport`,
then `/memory` to confirm both profile and todo were captured, then `/quit`,
restart, and `/memory` again — it should still be there.

If trustcall's JSON-patch flow misbehaves on Gemini, switch providers with one
line in `.env` (`AIMAESTRO_MODEL=openai:gpt-4o`) and install the extra:
`uv pip install --python .venv/bin/python -e ".[openai]"`. No code changes.

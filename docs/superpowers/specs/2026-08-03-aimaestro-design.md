# aiMaestro — Design

**Date:** 2026-08-03
**Status:** Approved, ready for implementation planning

## Summary

aiMaestro is a long-term-memory task assistant. It chats with you, and as it
chats it maintains three kinds of durable memory about you: a **profile** (who
you are), a **ToDo list** (what you need to do), and **instructions** (how you
like your ToDo list managed, learned from your feedback rather than configured).

The agent decides for itself when a message is worth remembering, and updates
memory incrementally — patching existing records instead of rewriting them.

## Goals

- A working assistant runnable from the terminal, whose memory survives restarts.
- Swappable LLM provider, set by one environment variable.
- A test suite that runs with no API key and costs nothing.
- A clean, self-contained public repo.

## Non-Goals

- Multi-user auth, hosting, or a server deployment.
- A web UI. (CLI plus LangGraph Studio only.)
- Vector/semantic memory search. Namespaced key lookup is sufficient at this scale.

## Architecture

Built on LangGraph. A single graph with one conversational node and three
memory-writing nodes:

```
START → aimaestro ──(no tool call)──→ END
             │
             ├─ update_profile ──────┐
             ├─ update_todos ────────┼─→ back to aimaestro
             └─ update_instructions ─┘
```

The `aimaestro` node loads all three memory namespaces from the store, injects
them into its system prompt, and replies. It is bound to a single `UpdateMemory`
tool whose only argument is `update_type: 'user' | 'todo' | 'instructions'`.
Calling that tool is how the model signals "this is worth saving." A conditional
edge routes on `update_type` to the matching writer node; each writer persists to
the store and returns a `ToolMessage`, and control returns to the conversational
node so it can respond naturally with the write already done.

Memory writes use **trustcall**, which asks the model for JSON-patch operations
against existing records rather than full rewrites. This is what keeps a long
ToDo list from being silently mangled on every update.

### Store layout

Namespaced by user, so multiple profiles coexist in one database:

| Namespace | Key | Value |
|---|---|---|
| `("profile", user_id)` | uuid | `Profile` |
| `("todo", user_id)` | uuid per task | `ToDo` |
| `("instructions", user_id)` | `user_instructions` | `{"memory": str}` |

## Components

| Module | Responsibility |
|---|---|
| `aimaestro/config.py` | Env-driven settings: model id, db path, user id. |
| `aimaestro/schemas.py` | `Profile`, `ToDo`, `UpdateMemory` pydantic models. |
| `aimaestro/prompts.py` | All prompt text, isolated so it can be tuned without touching logic. |
| `aimaestro/store.py` | Opens `SqliteStore` + `SqliteSaver`; owns the schema/migration concern. |
| `aimaestro/graph.py` | Nodes, routing, graph construction. |
| `aimaestro/cli.py` | Terminal chat loop and slash commands. |

`graph.py` exposes two entry points, because the graph is consumed two ways:

- `builder` / module-level `graph = builder.compile()` — for LangGraph Studio,
  which injects its own checkpointer and store at runtime.
- `build_graph(persist=True)` — compiles with `SqliteSaver` + `SqliteStore` for
  standalone CLI use.

This split is the fix for the central defect in the original code, which called
`builder.compile()` with neither backend and therefore only ran inside Studio.

## Model configuration

A single env var, `AIMAESTRO_MODEL`, resolved through
`langchain.chat_models.init_chat_model`:

```
AIMAESTRO_MODEL=google_genai:gemini-2.5-flash   # default
AIMAESTRO_MODEL=openai:gpt-4o
AIMAESTRO_MODEL=anthropic:claude-sonnet-4-5
```

Provider switching requires no code change. `temperature=0`, since every call is
either extraction or task reasoning.

## Persistence

`SqliteStore` for long-term memory, `SqliteSaver` for conversation threads, both
in `./data/aimaestro.db` (gitignored). Verified available in langgraph 1.2.10 and
langgraph-checkpoint-sqlite 3.1.1.

## CLI

```
python -m aimaestro [--user-id NAME] [--thread ID]
```

Slash commands: `/memory` prints current profile, todos, and learned
instructions; `/reset` clears the conversation thread but preserves long-term
memory; `/quit`.

## Error handling

- **Missing API key** — detected at startup with a message naming the env var to
  set and where to get a key, rather than a stack trace on first message.
- **Failed memory extraction** — the writer node logs and returns a `ToolMessage`
  noting the failure; the conversation continues. A bad extraction must never
  crash the chat.
- **Malformed `update_type`** — routed to `END` rather than raising, so an
  unexpected tool argument cannot kill the session.
- **Corrupt/locked database** — surfaced at startup with the db path.

## Testing

The suite runs with **no API key**. A `FakeToolCallingModel` returns scripted
tool calls, letting tests assert on graph behavior deterministically:

- Routing: each `update_type` reaches its correct writer node; absent tool calls
  terminate.
- Store writes: profile, todo, and instruction writes land in the right
  namespace with the right shape.
- **Persistence across restart** — write memory, dispose the graph, rebuild from
  the same db file, confirm memory is still there. This is the test that proves
  the original defect is fixed.
- Config resolution: env var precedence and defaults.
- Malformed routing degrades to `END` instead of raising.

One optional live smoke test, skipped unless a real key is present.

## Repo contents

```
aimaestro/           package
tests/               pytest suite
data/                sqlite db (gitignored)
langgraph.json       Studio registration for the single `aimaestro` graph
pyproject.toml       deps, pinned to Python 3.12
README.md  .env.example  .gitignore  LICENSE
```

The four course notebooks are **not** included; they stay in the local Desktop
copy as personal reference.

`LICENSE` is MIT and retains the upstream `Copyright (c) 2026 LangChain, Inc.`
line, with the user's copyright added alongside. The schemas, prompts, and graph
topology are derived work, so retaining that notice is the condition MIT attaches
to redistribution. No other course references appear anywhere in the repo.

## Environment

Python **3.12** via `uv` (system Python is 3.14, ahead of what this stack is
tested against; the upstream project pins 3.11).

## Known cleanups from the original code

- `builder.compile()` called with no checkpointer or store — the reason it could
  not run outside Studio.
- `ToDo.solutions` declares `min_items=1` alongside `default_factory=list`, which
  are contradictory; `min_items` is also deprecated in pydantic v2.
- `MemorySaver` imported but never used.
- Module-level model instantiation, which makes the import fail without a key
  rather than failing at first use with a clear message.

## Open item

No working API key exists on this machine yet — the `GOOGLE_API_KEY` in
`~/Desktop/to_do_agent/.env` is the literal placeholder `your-google-api-key-here`.
Everything ships and is fully tested without one. Obtaining a free Gemini key at
aistudio.google.com is the final step, documented in the README.

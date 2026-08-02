# aiMaestro

A task assistant that actually remembers you.

Most chat assistants forget everything the moment the window closes. aiMaestro
keeps three kinds of long-term memory, and decides on its own what is worth
writing down:

- **Profile** — who you are, where you live, what you care about.
- **ToDo list** — what you need to do, with deadlines and concrete next steps.
- **Instructions** — how *you* like your list managed, learned from your
  feedback rather than configured in a settings file.

Tell it something once and it holds on to it. Close the terminal, come back
tomorrow, and it still knows.

```
you › I'm Kratika, I live in Atlanta, and I need to renew my passport before October

aiMaestro › Added "renew passport" to your list with an October deadline.
            Since you're in Atlanta, the Atlanta Passport Agency handles
            expedited appointments if it gets tight.

you › /memory

Profile:
  name: Kratika
  location: Atlanta

ToDo:
  [not started] renew passport (by 2026-10-01)
      - Book an appointment at the Atlanta Passport Agency
      - Get passport photos taken
```

## How it works

One graph, four nodes. The conversational node answers you with everything it
already knows loaded into context, and it has exactly one tool available —
`UpdateMemory` — which it calls when something in your message deserves to be
remembered. That call routes to whichever writer node handles that kind of
memory, the write happens, and control returns so the reply lands with the work
already done.

```
START → aimaestro ──(nothing to save)──→ END
             │
             ├─ update_profile ──────┐
             ├─ update_todos ────────┼─→ back to aimaestro
             └─ update_instructions ─┘
```

Memory updates go through [trustcall](https://github.com/hinthornw/trustcall),
which asks the model for JSON-patch operations against existing records instead
of regenerating them wholesale. That distinction matters: it is what stops a
twelve-item ToDo list from being quietly mangled every time you add a thirteenth.

Everything is namespaced by user, so one database can hold several separate
memory profiles:

| Namespace | Holds |
|---|---|
| `("profile", user_id)` | A single `Profile` record |
| `("todo", user_id)` | One record per task |
| `("instructions", user_id)` | Your learned preferences |

## Quickstart

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/krarox22/aimaestro.git
cd aimaestro

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

cp .env.example .env      # then paste in your API key
.venv/bin/python -m aimaestro
```

You need one API key. A free Gemini key from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) is enough to run
everything here.

### Commands

| Command | Effect |
|---|---|
| `/memory` | Print everything currently remembered about you |
| `/reset` | Start a fresh conversation, keeping long-term memory |
| `/quit` | Exit |

```bash
python -m aimaestro --user-id kratika    # a separate memory profile
python -m aimaestro --thread abc123      # resume a past conversation
```

## Choosing a model

One line in `.env` switches providers — no code changes:

```bash
AIMAESTRO_MODEL=google_genai:gemini-2.5-flash   # default
AIMAESTRO_MODEL=openai:gpt-4o
AIMAESTRO_MODEL=anthropic:claude-sonnet-4-5
```

Install the matching extra for the non-default providers:

```bash
uv pip install --python .venv/bin/python -e ".[openai]"
uv pip install --python .venv/bin/python -e ".[anthropic]"
```

## Where memory lives

A single SQLite file at `data/aimaestro.db`, holding both the long-term store
and the conversation checkpoints. It is gitignored — your memory stays yours.
Delete the file to start over completely.

## Tests

```bash
.venv/bin/pytest -v
```

The suite runs against a scripted stand-in model, so it needs **no API key and
makes no network calls**. It covers routing, persistence, per-user isolation,
graceful degradation when extraction fails, and — most importantly — that memory
written in one session is still there in the next.

## LangGraph Studio

```bash
uv pip install --python .venv/bin/python "langgraph-cli[inmem]"
.venv/bin/langgraph dev
```

Opens the graph in Studio, where you can step through a turn node by node and
watch the store change.

## Project layout

```
aimaestro/
  config.py     env-driven settings and API key validation
  schemas.py    Profile, ToDo, UpdateMemory
  prompts.py    all prompt text, isolated from logic
  store.py      SQLite store + checkpointer
  graph.py      nodes, routing, graph construction
  cli.py        terminal chat interface
tests/          runs without an API key
```

## License

MIT.

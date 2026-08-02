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

## Evaluation

Tests prove the plumbing is correct. They cannot tell you whether the *model* is
any good at deciding what to remember. That is what `evals/` is for.

```bash
.venv/bin/python -m evals                      # the whole dataset, 3 runs each
.venv/bin/python -m evals --layer integrity    # just one layer
.venv/bin/python -m evals --scenario routing-ignores-small-talk --repeat 5
```

This one **does** call a real model and spend tokens.

The unit of evaluation is a *scenario*, not a message — a memory agent can't be
judged one turn at a time. Each scenario is a starting memory state, a sequence
of turns, and expectations about the memory state that results:

```yaml
id: integrity-completing-one-leaves-rest-intact
layer: integrity
seed:
  todo:
    - {task: vet appointment, status: not started}
    - {task: renew passport, status: not started}
turns:
  - the vet appointment is done
expect:
  changed_records: 1
  unchanged: [renew passport]
  field_values:
    vet appointment: {status: done}
```

Three layers, all scored **exactly** — no LLM judge anywhere:

| Layer | Question it answers |
|---|---|
| `routing` | Did it decide to save, and pick the right memory type? |
| `integrity` | Did updating one record leave every other record untouched? |
| `retrieval` | Did stored memory actually reach the prompt — and only this user's? |

`integrity` is the one worth watching. Patch-based memory updates exist to stop a
long list being rewritten every time one item changes, and this layer asserts the
bystanders come back byte-identical. `routing` is where over-saving shows up:
several scenarios expect *no* save, because the system prompt deliberately biases
toward capturing tasks and you want to know what that costs.

Results are **pass rates, not booleans**:

```
INTEGRITY
  FAIL integrity-completing-one-leaves-rest-intact   1/3   33%
         └─ 'renew passport' was modified: {...} -> {...}
  ok   integrity-adding-one-preserves-existing       3/3  100%
```

Tool-calling isn't deterministic even at `temperature=0`, so a single run tells
you little. Default is 3 repeats; raise it with `--repeat` when a result looks
marginal. `--threshold 0.8` exits non-zero below 80%, if you ever want this
gating something.

Adding a scenario means adding a YAML file — the loader rejects unknown keys, so
a typo in an expectation fails loudly instead of silently scoring as a pass.

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
evals/
  scenarios.py  scenario schema and loading
  runner.py     seeds memory, replays turns, captures the outcome
  scoring.py    the three layer scorers
  report.py     pass-rate aggregation and rendering
  scenarios/    the dataset, as YAML
tests/          runs without an API key
```

## License

MIT.

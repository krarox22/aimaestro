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
    """Render everything remembered about a user, for the /memory command."""
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
    parser = argparse.ArgumentParser(
        prog="aimaestro", description="A task assistant with long-term memory."
    )
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

            config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
            try:
                result = app.invoke({"messages": [("user", user_input)]}, config)
            except Exception as exc:
                print(f"\n[error] {exc}\n", file=sys.stderr)
                continue
            print(f"\naiMaestro › {result['messages'][-1].content}\n")


if __name__ == "__main__":
    raise SystemExit(main())

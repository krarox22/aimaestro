"""Run the evaluation harness: `python -m evals`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from aimaestro.config import Configuration, api_key_error
from evals.report import aggregate, overall, render
from evals.runner import run_scenario
from evals.scenarios import ScenarioError, load_scenarios

DEFAULT_SCENARIO_DIR = Path(__file__).parent / "scenarios"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="evals", description="Score aiMaestro's memory behavior."
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Runs per scenario. Tool-calling is not deterministic, so >1 "
        "gives a pass rate rather than a coin flip. Default 3.",
    )
    parser.add_argument(
        "--layer",
        choices=("routing", "integrity", "retrieval"),
        help="Only run scenarios of this layer.",
    )
    parser.add_argument("--scenario", help="Only run the scenario with this id.")
    parser.add_argument("--model", help="Override AIMAESTRO_MODEL for this run.")
    parser.add_argument(
        "--scenario-dir",
        default=str(DEFAULT_SCENARIO_DIR),
        help="Where scenario YAML files live.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Exit non-zero if the overall pass rate falls below this (0-1).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv()
    args = _parse_args(argv)

    cfg = Configuration.from_runnable_config(None)
    model_id = args.model or cfg.model

    problem = api_key_error(model_id)
    if problem:
        print(f"Cannot run evals: {problem}", file=sys.stderr)
        return 1

    try:
        scenarios = load_scenarios(args.scenario_dir)
    except ScenarioError as exc:
        print(f"Bad scenario: {exc}", file=sys.stderr)
        return 1

    if args.layer:
        scenarios = [s for s in scenarios if s.layer == args.layer]
    if args.scenario:
        scenarios = [s for s in scenarios if s.id == args.scenario]

    if not scenarios:
        print("No scenarios matched.", file=sys.stderr)
        return 1

    print(f"model: {model_id}")
    print(f"scenarios: {len(scenarios)}   repeats: {args.repeat}")

    results = []
    for scenario in scenarios:
        print(f"  running {scenario.id} ...", flush=True)
        runs = [
            run_scenario(scenario, model_id=model_id) for _ in range(args.repeat)
        ]
        results.extend(aggregate(scenario, runs))

    print(render(results))

    passes, total = overall(results)
    rate = passes / total if total else 0.0
    if rate < args.threshold:
        print(
            f"FAILED: pass rate {rate:.0%} is below threshold {args.threshold:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

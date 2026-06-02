"""Command-line interface: ``benchmark-fragility analyze <csv> [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, Optional

from .core import analyze
from .io import load_matrix
from .robustness import by_datasets, by_models
from .thresholds import Thresholds


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("csv", help="score matrix CSV (rows=tasks, cols=models)")
    p.add_argument("--direction", choices=["higher_better", "lower_better"],
                   default="higher_better")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--tau-w", type=float, default=0.60, help="win-rate threshold")
    p.add_argument("--tau-d", type=float, default=0.20, help="Cohen's d threshold")
    p.add_argument("--tau-b", type=float, default=0.20, help="breakdown threshold")
    p.add_argument("--json", action="store_true", help="emit JSON only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-fragility", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_common(sub.add_parser("analyze", help="per-pair diagnostics + summary"))
    rb = sub.add_parser("robustness", help="fragility by #models and #datasets")
    _add_common(rb)
    rb.add_argument("--seed", type=int, default=42)
    return parser


def _thresholds(args) -> Thresholds:
    return Thresholds(tau_w=args.tau_w, tau_d=args.tau_d, tau_b=args.tau_b)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = load_matrix(args.csv)
    th = _thresholds(args)

    if args.command == "analyze":
        res = analyze(matrix, direction=args.direction, top_k=args.top_k, thresholds=th)
        if args.json:
            print(json.dumps(res.summary, indent=2))
        else:
            s = res.summary
            print(f"models={s['n_models']} tasks={s['n_tasks']} pairs={s['n_pairs']} "
                  f"({s['direction']})")
            print(f"  magnitude   {s['magnitude']:6.2f}%")
            print(f"  consistency {s['consistency']:6.2f}%")
            print(f"  stability   {s['stability']:6.2f}%")
            print(f"  fragility   {s['fragility']:6.2f}%")
        return 0

    if args.command == "robustness":
        bm = by_models(matrix, direction=args.direction, thresholds=th)
        bd = by_datasets(matrix, seed=args.seed, top_k=args.top_k,
                         direction=args.direction, thresholds=th)
        if args.json:
            print(json.dumps({"by_models": bm.to_dict("records"),
                              "by_datasets": bd.to_dict("records")}, indent=2))
        else:
            print("By #models:\n" + bm.to_string(index=False))
            print("\nBy #datasets:\n" + bd.to_string(index=False))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

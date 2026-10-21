"""The coverage and critical-error curve of a system on dev and dev_real.

Run with `make threshold-report SYSTEM=...`. The threshold is chosen on dev, with the accuracy
target and the limit on critical errors of the gate, and then used as it is on dev_real: it is
not chosen again there. For each split the table shows, at a few thresholds and at the chosen
one, how many documents are automated and how many of them have a critical error, next to the
gate limits. The checks of today are applied to the saved answers and no model runs.

Neither split is part of the gate. With `--tune-on dev dev_real` the threshold is chosen on both
splits together, so no split is left over to check it on. With the default it is chosen on dev, and
dev_real is the check that it also holds on documents it was not chosen on.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adapt.config import Platform, load_platform
from adapt.data.rounds import resolve_round
from adapt.eval.rescore import rows_without_limit, tune_pooled
from adapt.eval.runner import DATA_DIR, load_documents, results_file
from adapt.eval.threshold import CurvePoint, Tuned, point_at

GRID = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
SPLITS = ("dev", "dev_real")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _check(point: CurvePoint, platform: Platform) -> str:
    gate = platform.config.gate
    parts = []
    for name, value, ok in (
        (
            "automation",
            point.automation_rate,
            point.automation_rate is not None and point.automation_rate >= gate.automation_rate_min,
        ),
        (
            "accuracy",
            point.accuracy_on_automated,
            point.accuracy_on_automated is not None
            and point.accuracy_on_automated >= gate.accuracy_automated_min,
        ),
        (
            "critical",
            point.critical_error_rate,
            point.critical_error_rate is not None
            and point.critical_error_rate <= gate.critical_error_rate_max,
        ),
    ):
        parts.append(f"{name} {_number(value)} {'pass' if ok else 'FAIL'}")
    return ", ".join(parts)


def format_split(
    split: str,
    count: int,
    rows: Sequence[dict[str, Any]],
    threshold: float,
    platform: Platform,
    note: str,
) -> str:
    thresholds = sorted({*GRID, threshold})
    lines = [
        f"{split} ({count} documents)",
        "  threshold  automated  automation  accuracy  critical errors  critical rate",
    ]
    for value in thresholds:
        point = point_at(rows, value)
        mark = f"  <- {note}" if value == threshold else ""
        lines.append(
            f"  {value:9.4f}  {point.automated:9d}  {_number(point.automation_rate):>10}"
            f"  {_number(point.accuracy_on_automated):>8}"
            f"  {point.critical_errors:>6} of {point.automated:<4d}"
            f"  {_number(point.critical_error_rate):>13}{mark}"
        )
    lines.append(f"  at {threshold:.4f}: {_check(point_at(rows, threshold), platform)}")
    return "\n".join(lines)


def describe(tuned: Tuned) -> str:
    if tuned.chosen is None:
        return "no threshold meets either target, so there is no confidence limit"
    if not tuned.critical_reached:
        return "the accuracy target is met, the limit on critical errors is not"
    if not tuned.accuracy_reached:
        return "the limit on critical errors is met, the accuracy target is not"
    return "meets the accuracy target and the limit on critical errors"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coverage and critical-error curve on dev and dev_real."
    )
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--system", required=True)
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--tune-on",
        nargs="+",
        choices=SPLITS,
        default=["dev"],
        help="the splits the threshold is chosen on together (default: dev)",
    )
    parser.add_argument("--in-dir", type=Path, help="default: results/round<N>/eval")
    args = parser.parse_args(argv)
    try:
        data_round = resolve_round(args.round, DATA_DIR)
    except ValueError as error:
        sys.exit(f"error: {error}")
    in_dir = args.in_dir or Path("results") / f"round{data_round}" / "eval"
    platform = load_platform(args.platform)
    gate = platform.config.gate

    saved = {}
    for split in dict.fromkeys([*args.tune_on, *args.splits]):
        path = results_file(in_dir, args.platform, args.system, split)
        if not path.exists():
            sys.exit(f"error: no saved {split} results for {args.system} in {in_dir}")
        saved[split] = (
            _load(path),
            load_documents(args.platform, split, DATA_DIR, data_round),
        )
    chosen_on = " and ".join(args.tune_on)
    try:
        tuned = tune_pooled([saved[name] for name in args.tune_on], platform)
    except ValueError as error:
        sys.exit(f"error: {chosen_on}: {error}")
    threshold = tuned.threshold if tuned.threshold is not None else 0.0
    print(f"threshold report for {args.system}, round {data_round}")
    print(
        f"limits: automation rate >= {gate.automation_rate_min:g}, accuracy on automated >= "
        f"{gate.accuracy_automated_min:g}, critical-error rate <= {gate.critical_error_rate_max:g}"
    )
    print(f"threshold chosen on {chosen_on}: {threshold:.4f}, {describe(tuned)}")
    for split in args.splits:
        results, documents = saved[split]
        rows = rows_without_limit(results, documents, platform)
        note = (
            f"chosen on {chosen_on}"
            if split in args.tune_on
            else f"from {chosen_on}, not chosen again"
        )
        print()
        print(format_split(split, len(documents), rows, threshold, platform, note))
    return 0


if __name__ == "__main__":
    sys.exit(main())

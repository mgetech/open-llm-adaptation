"""The eval gate: compare the results of a system with the thresholds in `platform.yaml`.

Run with `make gate SYSTEM=...`. Add `BASELINE=...` to include the test against the few-shot
system of the same size. A check whose results are missing counts as failed.

Where each number comes from:
- automation rate: the gold split;
- accuracy on automated documents and critical-error rate: gold and test_ood together;
- fine-tuned against few-shot: document accuracy on test_ood, McNemar exact test;
- generalisation gap: document accuracy on test_iid minus test_ood;
- review recall: the edge split;
- invalid JSON and invented accounts: all four splits together;
- injection success and canary leak: the worst of the four splits.
"""

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from adapt.config import Gate, load_platform
from adapt.eval.runner import RESULTS_DIR, results_file
from adapt.eval.stats import mcnemar_exact

GATE_SPLITS = ("gold", "test_iid", "test_ood", "edge")
Row = Mapping[str, Any]
Results = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Check:
    """One KPI against its limit. A value of None means there is no result to judge."""

    name: str
    value: float | None
    operator: Literal[">=", "<=", "<"]
    limit: float
    source: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        if self.value is None:
            return False
        # Rounded, so that 0.75 - 0.70 is 0.05 and not 0.05000000000000004.
        value = round(self.value, 9)
        if self.operator == ">=":
            return value >= self.limit
        if self.operator == "<=":
            return value <= self.limit
        return value < self.limit


def _missing(results: Results, splits: Sequence[str]) -> list[str]:
    return [split for split in splits if split not in results]


def _pooled(
    results: Results,
    splits: Sequence[str],
    numerator: Callable[[Row], bool],
    denominator: Callable[[Row], bool],
) -> float | None:
    """Sum of numerators over sum of denominators across the rows of several splits."""
    if _missing(results, splits):
        return None
    rows = [row for split in splits for row in results[split]["documents"]]
    count = sum(denominator(row) for row in rows)
    return sum(numerator(row) and denominator(row) for row in rows) / count if count else None


def _metric(results: Results, split: str, name: str) -> float | None:
    if split not in results:
        return None
    return results[split]["metrics"][name]["value"]


def _worst(results: Results, name: str) -> float | None:
    """The largest value of a metric over the gate splits. None if a split is missing."""
    if _missing(results, GATE_SPLITS):
        return None
    values = [_metric(results, split, name) for split in GATE_SPLITS]
    defined = [v for v in values if v is not None]
    return max(defined) if defined else None


def _where(results: Results, splits: Sequence[str], text: str) -> str:
    missing = _missing(results, splits)
    return f"not run: {', '.join(missing)}" if missing else text


def _automated(row: Row) -> bool:
    return not row["needs_review"]


def _reason(code: str) -> Callable[[Row], bool]:
    return lambda row: code in row["reasons"]


def _always(row: Row) -> bool:
    return True


def check_gate(
    gate: Gate,
    results: Results,
    baseline_test_ood: Mapping[str, Any] | None = None,
    baseline_note: str = "not run: no baseline given",
) -> list[Check]:
    """All checks, in the order of the KPI table. `results` maps a split to its results file.

    `baseline_test_ood` is the test_ood results of the few-shot system. Without it the test
    against few-shot is not run, and `baseline_note` says why."""
    both = ("gold", "test_ood")
    accuracy = _pooled(results, both, lambda r: r["correct"], _automated)
    critical = _pooled(results, both, lambda r: r["severity"] == "critical", _automated)

    iid = _metric(results, "test_iid", "document_accuracy")
    ood = _metric(results, "test_ood", "document_accuracy")
    gap = None if iid is None or ood is None else iid - ood

    p_value, detail = None, baseline_note
    if baseline_test_ood is not None and "test_ood" in results:
        test = mcnemar_exact(
            {row["id"]: row["correct"] for row in results["test_ood"]["documents"]},
            {row["id"]: row["correct"] for row in baseline_test_ood["documents"]},
        )
        detail = f"only the system right on {test.only_a}, only the baseline on {test.only_b}"
        # A system that is significantly worse than the baseline does not pass.
        p_value = test.p_value if test.only_a > test.only_b else 1.0
    elif baseline_test_ood is not None:
        detail = "not run: test_ood"

    def pooled_reason(code: str) -> float | None:
        return _pooled(results, GATE_SPLITS, _reason(code), _always)

    return [
        Check(
            "automation rate",
            _metric(results, "gold", "automation_rate"),
            ">=",
            gate.automation_rate_min,
            _where(results, ["gold"], "gold"),
        ),
        Check(
            "accuracy on automated documents",
            accuracy,
            ">=",
            gate.accuracy_automated_min,
            _where(results, both, "gold + test_ood"),
        ),
        Check(
            "critical-error rate",
            critical,
            "<=",
            gate.critical_error_rate_max,
            _where(results, both, "gold + test_ood"),
        ),
        Check(
            "fine-tuned vs few-shot, p-value",
            p_value,
            "<",
            gate.mcnemar_p_max,
            "test_ood",
            detail,
        ),
        Check(
            "generalisation gap",
            gap,
            "<=",
            gate.generalisation_gap_max,
            _where(results, ["test_iid", "test_ood"], "test_iid - test_ood"),
        ),
        Check(
            "review-flag recall",
            _metric(results, "edge", "review_recall"),
            ">=",
            gate.review_recall_min,
            _where(results, ["edge"], "edge"),
        ),
        Check(
            "invalid JSON rate",
            pooled_reason("invalid_output"),
            "<=",
            gate.invalid_json_rate_max,
            _where(results, GATE_SPLITS, "all four splits"),
        ),
        Check(
            "invented account rate",
            pooled_reason("unknown_account"),
            "<=",
            gate.invented_account_rate_max,
            _where(results, GATE_SPLITS, "all four splits"),
        ),
        Check(
            "prompt-injection success",
            _worst(results, "injection_success_rate"),
            "<=",
            gate.injection_success_max,
            _where(results, GATE_SPLITS, "worst of four splits"),
        ),
        Check(
            "canary leak",
            _worst(results, "canary_leak_rate"),
            "<=",
            gate.canary_leak_max,
            _where(results, GATE_SPLITS, "worst of four splits"),
        ),
    ]


def report_only(results: Results) -> list[tuple[str, str]]:
    """Numbers that are shown but not judged, taken from test_ood."""
    if "test_ood" not in results:
        return [("item F1, weighted error cost, speed", "not run: test_ood")]
    performance = results["test_ood"]["performance"]
    p50, p95 = performance["latency_p50_seconds"], performance["latency_p95_seconds"]
    cost = _metric(results, "test_ood", "weighted_error_cost")
    return [
        ("item F1", _format(_metric(results, "test_ood", "item_f1"))),
        ("weighted error cost per 1,000 documents", _format(cost)),
        ("documents per second", _format(performance["documents_per_second"])),
        ("latency p50 / p95 seconds", f"{_format(p50)} / {_format(p95)}"),
    ]


def rule_report(results: Results) -> list[tuple[str, str]]:
    """How many documents a pipeline rule flagged, and what that does to review recall.

    A rule is a document check (a missing total, an unsupported chart, a foreign currency). It
    flags a document without the model, so its flags are shown apart from the model's. Results
    from before the runs kept this have no `by_rule` field."""
    rows = [row for split in GATE_SPLITS if split in results for row in results[split]["documents"]]
    if not rows or any("by_rule" not in row for row in rows):
        return [("flagged by a pipeline rule", "not recorded in these results")]
    flagged = [row for row in rows if row["by_rule"]]
    currency = sum("foreign_currency" in row["reasons"] for row in flagged)
    shown = [
        (
            "documents flagged by a pipeline rule",
            f"{len(flagged)} of {len(rows)} ({len(flagged) / len(rows):.1%}), "
            f"{currency} of them for foreign currency",
        )
    ]
    wanted = [row for row in results.get("edge", {}).get("documents", []) if row["expected_review"]]
    if wanted:
        with_rule = sum(row["needs_review"] for row in wanted) / len(wanted)
        model_only = sum(row["needs_review"] and not row["by_rule"] for row in wanted) / len(wanted)
        shown.append(
            (
                "review recall on edge, with / without the rules",
                f"{_format(with_rule)} / {_format(model_only)}",
            )
        )
    return shown


def _format(value: float | None) -> str:
    return "no data" if value is None else f"{value:.3f}"


def format_table(checks: Sequence[Check]) -> str:
    """Each KPI with its value, its limit, where it comes from and whether it passed."""
    rows = [
        (
            check.name,
            _format(check.value),
            f"{check.operator} {check.limit:g}",
            "pass" if check.passed else "FAIL",
            "; ".join(part for part in (check.source, check.detail) if part),
        )
        for check in checks
    ]
    header = ("KPI", "value", "limit", "result", "from")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(4)]
    lines = []
    for row in [header, *rows]:
        cells = "  ".join(cell.ljust(width) for cell, width in zip(row[:4], widths, strict=True))
        lines.append(f"{cells}  {row[4]}".rstrip())
    return "\n".join(lines)


def load_results(out_dir: Path, platform: str, system: str) -> dict[str, dict[str, Any]]:
    """The results files of a system for the gate splits. Splits without a file are left out."""
    found = {}
    for split in GATE_SPLITS:
        path = results_file(out_dir, platform, system, split)
        if path.exists():
            found[split] = json.loads(path.read_text(encoding="utf-8"))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the results of a system against the gate.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--system", required=True)
    parser.add_argument("--baseline", help="few-shot system of the same size")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    gate = load_platform(args.platform).config.gate
    results = load_results(args.out_dir, args.platform, args.system)
    baseline = None
    note = "not run: no baseline given"
    if args.baseline:
        baseline_path = results_file(args.out_dir, args.platform, args.baseline, "test_ood")
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            note = f"not run: no test_ood results for {args.baseline}"
    checks = check_gate(gate, results, baseline, note)

    title = f"gate for {args.system}" + (f" against {args.baseline}" if args.baseline else "")
    print(title)
    print(format_table(checks))
    print()
    for name, value in [*report_only(results), *rule_report(results)]:
        print(f"report only: {name}: {value}")
    failed = [check for check in checks if not check.passed]
    print()
    if failed:
        print(f"gate: FAIL ({len(failed)} of {len(checks)} checks)")
        return 1
    print(f"gate: PASS ({len(checks)} of {len(checks)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

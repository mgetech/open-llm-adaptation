"""The gate checks that dev and dev_real can show, for every run.

Run with `make stage-gate SYSTEM=...`. The gate itself reads test_ood, the gold set and the other
test splits, and it is run once for the claim (`make gate`, with `FINAL = True` in the training
notebook). The stage report is the same checks on dev and dev_real, which are the splits used to
choose settings. Their numbers guide those choices and are not a claim. The threshold is the one
chosen on dev.

Two checks cannot be shown here: the generalisation gap needs test_iid and test_ood, and the
test against few-shot is on test_ood. Review-flag recall is the gate's check on the edge split;
here it is the recall on the documents of dev or dev_real that should be reviewed.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adapt.config import Gate, load_platform
from adapt.data.rounds import eval_dir
from adapt.eval.runner import results_file

STAGE_SPLITS = ("dev", "dev_real")
NOT_SHOWN = (
    "generalisation gap: needs test_iid and test_ood",
    "fine-tuned vs few-shot: needs the few-shot baseline on test_ood",
)


def checks(gate: Gate) -> list[tuple[str, str, str, float]]:
    """Name, metric, operator and limit of each check the stage report shows."""
    return [
        ("automation rate", "automation_rate", ">=", gate.automation_rate_min),
        (
            "accuracy on automated documents",
            "accuracy_on_automated",
            ">=",
            gate.accuracy_automated_min,
        ),
        ("critical-error rate", "critical_error_rate", "<=", gate.critical_error_rate_max),
        ("review-flag recall", "review_recall", ">=", gate.review_recall_min),
        ("invalid JSON rate", "invalid_json_rate", "<=", gate.invalid_json_rate_max),
        ("invented account rate", "invented_account_rate", "<=", gate.invented_account_rate_max),
        ("prompt-injection success", "injection_success_rate", "<=", gate.injection_success_max),
        ("canary leak", "canary_leak_rate", "<=", gate.canary_leak_max),
    ]


def verdict(value: float | None, operator: str, limit: float) -> str:
    if value is None:
        return "n/a"
    value = round(value, 9)
    ok = value >= limit if operator == ">=" else value <= limit
    return "pass" if ok else "FAIL"


def _value(results: dict[str, Any], metric: str) -> float | None:
    if metric == "injection_success_rate" and "injection_unexpected_rate" not in results["metrics"]:
        return None  # saved with the old definition of this figure
    return results["metrics"].get(metric, {}).get("value")


def _counts(results: dict[str, Any]) -> str:
    rows = results["documents"]
    automated = [row for row in rows if not row["needs_review"]]
    critical = sum(row["severity"] == "critical" for row in automated)
    return f"{critical} critical in {len(automated)} automated of {len(rows)}"


def table(gate: Gate, results: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """The report as text, and the names of the checks that failed on any split."""
    splits = list(results)
    header = ["check", "limit", *(f"{s}" for s in splits)]
    rows = [header]
    failed: list[str] = []
    for name, metric, operator, limit in checks(gate):
        line = [name, f"{operator} {limit:g}"]
        for split in splits:
            value = _value(results[split], metric)
            result = verdict(value, operator, limit)
            shown = "no documents" if value is None and result == "n/a" else f"{value:.4f}"
            line.append(f"{shown} {result}")
            if result == "FAIL" and name not in failed:
                failed.append(name)
        rows.append(line)
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    text = "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    )
    return text, failed


def report(gate: Gate, results: dict[str, dict[str, Any]], system: str) -> tuple[str, bool]:
    """The whole stage report and whether every check that could be judged passed."""
    text, failed = table(gate, results)
    threshold = next(iter(results.values()))["manifest"]["threshold"]
    source = results.get("dev", next(iter(results.values())))["confidence_threshold"]["source"]
    lines = [
        f"stage report for {system}: {' and '.join(results)}, not the claim",
        f"confidence threshold {threshold:.4f} ({source})",
        "",
        text,
        "",
        *(
            f"{split}: {_counts(data)}, document accuracy "
            f"{data['metrics']['document_accuracy']['value']:.3f}"
            for split, data in results.items()
        ),
        "not shown: " + "; ".join(NOT_SHOWN),
        "",
        "stage: FAIL (" + ", ".join(failed) + ")" if failed else "stage: PASS (every check shown)",
    ]
    return "\n".join(lines), not failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate checks on dev and dev_real.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--system", required=True)
    parser.add_argument("--round", type=int, help="round of the results (default: the current one)")
    parser.add_argument("--out-dir", type=Path, help="default: results/round<N>/eval")
    args = parser.parse_args(argv)
    try:
        out_dir = args.out_dir or eval_dir(args.round)
    except ValueError as error:
        sys.exit(f"error: {error}")
    results = {}
    for split in STAGE_SPLITS:
        path = results_file(out_dir, args.platform, args.system, split)
        if path.exists():
            results[split] = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        sys.exit(f"error: no dev or dev_real results for {args.system} in {out_dir}")
    text, passed = report(load_platform(args.platform).config.gate, results, args.system)
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

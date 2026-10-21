"""A post-hoc re-scoring of saved results with the changes made after the claim run.

Run with `make post-hoc SYSTEM=... BASELINE=... ROUND=3`. It reads the saved results of the
claim run, scores the saved answers again with today's checks and the injection figure of today,
and uses a confidence threshold chosen on dev only, with the accuracy target and the limit on
critical errors. No model runs. The gate is then worked out on the re-scored splits.

This is not a claim. The test splits and gold_v2 it reads were evaluated in the claim run, and
the changes were made after that result was known. The threshold is never chosen or changed with
the test splits, and nothing is chosen from this report. The report is written to
`results/post_hoc/round<N>_rescored.json` and says so.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adapt.config import Platform, load_platform
from adapt.data.rounds import eval_dir, resolve_round
from adapt.eval.gate import (
    check_gate,
    format_table,
    injection_report,
    report_only,
    rule_report,
)
from adapt.eval.rescore import rescore, tune_on_dev
from adapt.eval.runner import DATA_DIR, ROOT, git_state, hash_file, load_documents, results_file
from adapt.eval.threshold import tuning_record

SPLITS = ("dev", "test_iid", "test_ood", "edge", "gold", "gold_v2")
GOLD_SETS = ("gold_v2", "gold")
BASELINE_SPLITS = ("dev", "test_ood")
LABEL = (
    "POST-HOC, NOT A CLAIM. The saved outputs of the round 3 claim run, scored again with the "
    "injection figure and the confidence threshold chosen after that run. The test splits and "
    "gold_v2 were already evaluated in the claim run, so they cannot back another claim. The "
    "threshold was chosen on dev only and was not changed after looking at the other splits."
)
RESULT_DIR = ROOT / "results" / "post_hoc"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rescored(
    in_dir: Path,
    platform_name: str,
    platform: Platform,
    system: str,
    splits: Sequence[str],
    data_round: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], Any]:
    """The saved results of `system` scored again, the hashes of the files read, and the tuning.

    The threshold is chosen on dev. Every other split gets that threshold as it is."""
    saved: dict[str, dict[str, Any]] = {}
    inputs: dict[str, str] = {}
    for split in splits:
        path = results_file(in_dir, platform_name, system, split)
        saved[split] = _load(path)
        inputs[f"{system}/{split}"] = hash_file(path)
    documents = {s: load_documents(platform_name, s, DATA_DIR, data_round) for s in splits}
    tuned = tune_on_dev(saved["dev"], documents["dev"], platform)
    done = {}
    for split in splits:
        done[split], _ = rescore(saved[split], documents[split], platform, tuned, split)
    return done, inputs, tuned


def _checks(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": c.name,
            "value": c.value,
            "operator": c.operator,
            "limit": c.limit,
            "passed": c.passed,
            "source": c.source,
            "detail": c.detail,
        }
        for c in rows
    ]


def build(
    platform_name: str, system: str, baseline: str, data_round: int, in_dir: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    """The post-hoc report as a dictionary, and the text of the gate tables for each gold set."""
    platform = load_platform(platform_name)
    done, inputs, tuned = _rescored(in_dir, platform_name, platform, system, SPLITS, data_round)
    base, base_inputs, base_tuned = _rescored(
        in_dir, platform_name, platform, baseline, BASELINE_SPLITS, data_round
    )
    inputs |= base_inputs
    gate = platform.config.gate
    gates: dict[str, Any] = {}
    texts: dict[str, str] = {}
    for gold in GOLD_SETS:
        checks = check_gate(gate, done, base["test_ood"], "not run: no baseline given", gold)
        passed = sum(c.passed for c in checks)
        gates[gold] = {
            "gold_set": gold,
            "passed": passed,
            "total": len(checks),
            "result": "PASS" if passed == len(checks) else "FAIL",
            "checks": _checks(checks),
            "report_only": [
                list(row)
                for row in (
                    *report_only(done),
                    *injection_report(done, gold),
                    *rule_report(done, gold),
                )
            ],
        }
        texts[gold] = format_table(checks)
    commit, dirty = git_state()
    report = {
        "post_hoc": True,
        "claim": False,
        "label": LABEL,
        "round": data_round,
        "system": system,
        "baseline": baseline,
        "git_commit": commit,
        "git_dirty": dirty,
        "threshold": {
            "value": tuned.threshold,
            "chosen_on": "dev",
            "rule": "lowest threshold with accuracy on automated documents at or above "
            f"{gate.accuracy_automated_min:g} and critical-error rate at or below "
            f"{gate.critical_error_rate_max:g}",
            "both_targets_reached": tuned.chosen is not None and tuned.critical_reached,
            "dev": tuning_record(tuned),
            "baseline": {
                "value": base_tuned.threshold if base_tuned.threshold is not None else 0.0,
                "chosen_on": "dev",
                "accuracy_target_reached": base_tuned.threshold is not None,
            },
        },
        "inputs_sha256": inputs,
        "gates": gates,
        "splits": {
            split: {
                "documents": len(data["documents"]),
                "threshold": data["manifest"]["threshold"],
                "metrics": {
                    name: data["metrics"][name]["value"]
                    for name in (
                        "document_accuracy",
                        "automation_rate",
                        "accuracy_on_automated",
                        "critical_error_rate",
                        "review_recall",
                        "injection_success_rate",
                        "injection_unexpected_rate",
                        "canary_leak_rate",
                    )
                },
                "severity_counts": data["metrics"]["severity_counts"],
            }
            for split, data in done.items()
        },
    }
    return report, texts


def format_report(report: dict[str, Any], texts: dict[str, str]) -> str:
    lines = [
        report["label"],
        "",
        f"{report['system']} against {report['baseline']}, round {report['round']}",
    ]
    threshold = report["threshold"]
    lines.append(
        f"confidence threshold {threshold['value']:.4f}, chosen on dev: {threshold['rule']}"
    )
    for gold, gate in report["gates"].items():
        lines += ["", f"post-hoc gate with {gold} as the gold set", texts[gold], ""]
        lines.append(
            f"post-hoc gate: {gate['result']} ({gate['passed']} of {gate['total']} checks)"
        )
        lines += [f"report only: {name}: {value}" for name, value in gate["report_only"]]
    lines += ["", "POST-HOC, NOT A CLAIM."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-hoc re-scoring of a claim run.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--system", required=True)
    parser.add_argument("--baseline", required=True, help="few-shot system of the same size")
    parser.add_argument("--round", type=int, help="round of the results (default: the current one)")
    parser.add_argument("--in-dir", type=Path, help="default: results/round<N>/eval")
    parser.add_argument("--out", type=Path, help="default: results/post_hoc/round<N>_rescored.json")
    args = parser.parse_args(argv)
    try:
        data_round = resolve_round(args.round, DATA_DIR)
        in_dir = args.in_dir or eval_dir(data_round)
        report, texts = build(args.platform, args.system, args.baseline, data_round, in_dir)
    except (ValueError, FileNotFoundError) as error:
        sys.exit(f"error: {error}")
    out = args.out or RESULT_DIR / f"round{data_round}_rescored.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(format_report(report, texts))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

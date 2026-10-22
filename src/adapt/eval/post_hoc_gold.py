"""A post-hoc re-scoring of the shipped configuration on gold and gold_v2.

Run with `make post-hoc-gold SYSTEM=... ROUND=3`. It scores the saved outputs of gold and gold_v2
again with every pipeline check of today and the confidence threshold of the shipped
configuration, which is chosen on dev and dev_real together. No model runs, and nothing is
chosen from this report. gold_v3 is not read: only gold and gold_v2 are accepted.

This is not a claim. Both sets were evaluated in earlier runs, and some of the checks were
written after I saw those results. The report is written to `results/post_hoc/gold_rescored.json`
and says so.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adapt.config import load_platform
from adapt.data.rounds import eval_dir, resolve_round
from adapt.eval.rescore import rescore, tune_pooled
from adapt.eval.runner import DATA_DIR, ROOT, git_state, hash_file, load_documents, results_file

GOLD_SETS = ("gold", "gold_v2")
TUNE_ON = ("dev", "dev_real")
METRICS = (
    "document_accuracy",
    "automation_rate",
    "accuracy_on_automated",
    "critical_error_rate",
    "review_recall",
    "injection_success_rate",
    "injection_unexpected_rate",
    "canary_leak_rate",
)
LABEL = (
    "POST-HOC, NOT A CLAIM. The saved outputs of the shipped configuration on gold and gold_v2, "
    "scored again with every pipeline check of today and a confidence threshold chosen on dev "
    "and dev_real together. Both sets were evaluated before, and some of the checks were written "
    "after I saw those results, so these numbers are not held out. Nothing was changed because "
    "of them."
)
OUT = ROOT / "results" / "post_hoc" / "gold_rescored.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(platform_name: str, system: str, data_round: int, in_dir: Path) -> dict[str, Any]:
    platform = load_platform(platform_name)
    inputs: dict[str, str] = {}
    saved: dict[str, dict[str, Any]] = {}
    for split in (*TUNE_ON, *GOLD_SETS):
        path = results_file(in_dir, platform_name, system, split)
        saved[split] = _load(path)
        inputs[f"{system}/{split}"] = hash_file(path)
    documents = {s: load_documents(platform_name, s, DATA_DIR, data_round) for s in saved}
    tuned = tune_pooled([(saved[s], documents[s]) for s in TUNE_ON], platform)
    if tuned.threshold is None:
        raise ValueError("no threshold meets the targets on dev and dev_real")
    splits = {}
    for gold in GOLD_SETS:
        done, _ = rescore(saved[gold], documents[gold], platform, tuned, gold, TUNE_ON)
        rows = done["documents"]
        automated = [row for row in rows if not row["needs_review"]]
        splits[gold] = {
            "documents": len(rows),
            "automatable": sum(not row["expected_review"] for row in rows),
            "automated": len(automated),
            "automated_correct": sum(bool(row["correct"]) for row in automated),
            "critical_errors": sum(row["severity"] == "critical" for row in automated),
            "metrics": {name: done["metrics"][name]["value"] for name in METRICS},
            "severity_counts": done["metrics"]["severity_counts"],
        }
    commit, dirty = git_state()
    return {
        "post_hoc": True,
        "claim": False,
        "label": LABEL,
        "round": data_round,
        "system": system,
        "git_commit": commit,
        "git_dirty": dirty,
        "threshold": {
            "value": tuned.threshold,
            "chosen_on": " and ".join(TUNE_ON),
            "rule": "the limit on critical errors first, then the accuracy target on automated "
            "documents",
        },
        "inputs_sha256": inputs,
        "splits": splits,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        report["label"],
        "",
        f"{report['system']}, round {report['round']}, threshold "
        f"{report['threshold']['value']:.4f} (chosen on {report['threshold']['chosen_on']})",
    ]
    for gold, data in report["splits"].items():
        m = data["metrics"]
        lines += [
            "",
            f"{gold}: {data['documents']} documents",
            f"  automation rate {m['automation_rate']:.3f} "
            f"({data['automated']} automated of {data['automatable']} that need no review)",
            f"  accuracy on automated documents {m['accuracy_on_automated']:.3f}",
            f"  critical errors {data['critical_errors']} of {data['automated']} automated",
            f"  review recall {m['review_recall']:.3f}",
        ]
    lines += ["", "POST-HOC, NOT A CLAIM."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-hoc re-scoring on gold and gold_v2.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--system", required=True)
    parser.add_argument("--round", type=int, help="round of the results (default: the current one)")
    parser.add_argument("--in-dir", type=Path, help="default: results/round<N>/eval")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    try:
        data_round = resolve_round(args.round, DATA_DIR)
        in_dir = args.in_dir or eval_dir(data_round)
        report = build(args.platform, args.system, data_round, in_dir)
    except (ValueError, FileNotFoundError) as error:
        sys.exit(f"error: {error}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(format_report(report))
    print(f"\nreport: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

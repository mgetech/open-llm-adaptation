"""Compare two results files of one split: the same adapter run on another machine.

Run with `make compare FIRST=... SECOND=...`. The first file is the reference (the Kaggle run), the
second is the new one (the VM). The check passes when both ran the same base model revision,
adapter, prompt and confidence threshold, they hold the same documents, and every metric of the
second run is inside the confidence interval of the first. It also counts the documents whose
answer is identical. The GPU, the vLLM version and the commit may differ, and they are listed.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

METRICS = (
    "document_accuracy",
    "automation_rate",
    "accuracy_on_automated",
    "critical_error_rate",
    "review_recall",
)
MUST_MATCH = ("model_revision", "adapter_hash", "prompt_sha256", "threshold", "few_shot")
MAY_DIFFER = ("gpu", "vllm_version", "git_commit")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """The comparison of two results files, as a dict that `main` prints."""
    one, two = first["manifest"], second["manifest"]
    manifest = {
        key: {"first": one.get(key), "second": two.get(key), "same": one.get(key) == two.get(key)}
        for key in (*MUST_MATCH, *MAY_DIFFER)
    }
    rows_one = {row["id"]: row for row in first["documents"]}
    rows_two = {row["id"]: row for row in second["documents"]}
    shared = sorted(rows_one.keys() & rows_two.keys())
    identical = sum(rows_one[i]["output"] == rows_two[i]["output"] for i in shared)
    same_decision = sum(rows_one[i]["needs_review"] == rows_two[i]["needs_review"] for i in shared)
    gaps = [
        abs(rows_one[i]["confidence"] - rows_two[i]["confidence"])
        for i in shared
        if rows_one[i].get("confidence") is not None and rows_two[i].get("confidence") is not None
    ]
    metrics = {}
    for name in METRICS:
        a, b = first["metrics"].get(name), second["metrics"].get(name)
        if not a or not b or a["value"] is None or b["value"] is None:
            continue
        metrics[name] = {
            "first": a["value"],
            "second": b["value"],
            "interval": [a["low"], a["high"]],
            "inside": a["low"] <= b["value"] <= a["high"],
        }
    same_documents = rows_one.keys() == rows_two.keys()
    passed = (
        same_documents
        and all(manifest[key]["same"] for key in MUST_MATCH)
        and bool(metrics)
        and all(metric["inside"] for metric in metrics.values())
    )
    return {
        "split": first["split"],
        "same_split": first["split"] == second["split"],
        "same_documents": same_documents,
        "documents": len(shared),
        "identical_outputs": identical,
        "same_review_decision": same_decision,
        "largest_confidence_difference": max(gaps, default=None),
        "manifest": manifest,
        "metrics": metrics,
        "passed": passed and first["split"] == second["split"],
    }


def render(report: Mapping[str, Any]) -> str:
    lines = [f"split {report['split']}: {report['documents']} documents"]
    lines.append(f"identical outputs: {report['identical_outputs']} of {report['documents']}")
    lines.append(f"same review decision: {report['same_review_decision']} of {report['documents']}")
    gap = report["largest_confidence_difference"]
    if gap is not None:
        lines.append(f"largest confidence difference: {gap:.6f}")
    for key, entry in report["manifest"].items():
        must = key in MUST_MATCH
        mark = "same" if entry["same"] else ("DIFFERENT" if must else "differs")
        lines.append(f"{key}: {mark} ({entry['first']} / {entry['second']})")
    for name, metric in report["metrics"].items():
        low, high = metric["interval"]
        mark = "inside" if metric["inside"] else "OUTSIDE"
        lines.append(
            f"{name}: {metric['first']:.4f} / {metric['second']:.4f}, interval {low:.4f} to "
            f"{high:.4f}, {mark}"
        )
    lines.append("PASS" if report["passed"] else "FAIL")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two results files of one split.")
    parser.add_argument("--first", type=Path, required=True, help="the reference results file")
    parser.add_argument("--second", type=Path, required=True, help="the new results file")
    parser.add_argument("--out", type=Path, help="also write the comparison as JSON")
    args = parser.parse_args(argv)
    try:
        report = compare(load(args.first), load(args.second))
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"error: {error!r}")
    print(render(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

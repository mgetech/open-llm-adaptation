"""A configuration of two adapters, worked out from two saved results files.

Run with `make pair FIRST=... SECOND=... NAME=...`. Two adapters answer the same documents. A
document is automated only if both answers are identical (the same items, the same
reverse-charge flag and the same total) and neither adapter sent it to review. Any other
document goes to review. Because the answers are saved at temperature 0, combining two results
files gives what a served pair would answer, and no model is run.

The combined results have the layout of any other system, so `make gate SYSTEM=<name>` reads them.
A single adapter is a configuration already and needs no file.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from adapt.config import load_platform
from adapt.data.rounds import eval_dir, resolve_round
from adapt.eval.metrics import account_groups
from adapt.eval.runner import (
    ALL_SPLITS,
    DATA_DIR,
    SYSTEM_NAME,
    git_state,
    load_documents,
    results_file,
    write_results,
)
from adapt.eval.scoring import EvalDocument, compute_metrics, compute_slices, document_rows, score
from adapt.eval.stats import SEED
from adapt.pipeline import Result

LOW_CONFIDENCE = "low_confidence"
DISAGREE = "adapters_disagree"
# `each`: every adapter keeps the confidence threshold tuned for it on dev.
# `ignore`: a low confidence does not send a document to review, only the agreement does.
Confidence = Literal["each", "ignore"]
Signature = tuple[tuple[tuple[str, str, float], ...], bool, float | None]


def answer_signature(output: Mapping[str, Any] | None) -> Signature | None:
    """What two answers must have in common: the items, the reverse-charge flag, the total."""
    if output is None:
        return None
    items = sorted(
        (item["account"], str(item["vat_rate"]), round(float(item["amount"]), 2))
        for item in output["items"]
    )
    total = output.get("total")
    return (
        tuple(items),
        bool(output.get("reverse_charge")),
        None if total is None else round(float(total), 2),
    )


def _reasons(row: Mapping[str, Any], confidence: Confidence) -> list[str]:
    reasons = list(row["reasons"])
    return [r for r in reasons if r != LOW_CONFIDENCE] if confidence == "ignore" else reasons


def _flagged(row: Mapping[str, Any], confidence: Confidence) -> bool:
    if confidence == "ignore":
        return bool(_reasons(row, confidence))
    return bool(row["needs_review"])


def combine_row(
    first: Mapping[str, Any], second: Mapping[str, Any], confidence: Confidence
) -> dict[str, Any]:
    """The outcome for one document: whether it is in review, why, and the answer kept."""
    members = (first, second)
    flagged = [m for m in members if _flagged(m, confidence)]
    reasons: list[str] = []
    for member in members:
        reasons += [r for r in _reasons(member, confidence) if r not in reasons]
    first_answer, second_answer = (
        answer_signature(first["output"]),
        answer_signature(second["output"]),
    )
    disagree = (
        first_answer is not None and second_answer is not None and (first_answer != second_answer)
    )
    if disagree:
        reasons.append(DISAGREE)
    needs_review = bool(flagged) or disagree
    confidences = [m["confidence"] for m in members if m["confidence"] is not None]
    return {
        "needs_review": needs_review,
        "reasons": reasons,
        "output": first["output"],
        # A flag is a rule flag if every adapter that flagged the document was flagged by a
        # rule, and the adapters did not also disagree.
        "by_rule": bool(flagged) and not disagree and all(m["by_rule"] for m in flagged),
        "confidence": min(confidences) if confidences else None,
    }


def _check_members(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if first["split"] != second["split"]:
        raise ValueError(
            f"the results are for different splits: {first['split']}, {second['split']}"
        )
    if first["manifest"].get("data_sha256") != second["manifest"].get("data_sha256"):
        raise ValueError("the two systems ran on different data")
    ids = [row["id"] for row in first["documents"]]
    if ids != [row["id"] for row in second["documents"]]:
        raise ValueError("the two systems have different documents")
    for results in (first, second):
        rows = results["documents"]
        if any("output" not in row or "by_rule" not in row for row in rows):
            raise ValueError(
                f"{results['system']} was run before the results kept the model output, "
                "so it cannot be combined"
            )


def combine_results(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    name: str,
    documents: Sequence[EvalDocument],
    groups: Mapping[str, str],
    confidence: Confidence = "each",
) -> dict[str, Any]:
    """The results of the pair on one split, in the layout of any other results file."""
    _check_members(first, second)
    if [d.id for d in documents] != [row["id"] for row in first["documents"]]:
        raise ValueError("the documents do not match the results")
    combined = [
        combine_row(a, b, confidence)
        for a, b in zip(first["documents"], second["documents"], strict=True)
    ]
    results = [
        Result(
            request_id=document.id,
            needs_review=row["needs_review"],
            reasons=row["reasons"],
            items=row["output"]["items"] if row["output"] else [],
            booking_lines=[],
            adapter=name,
            adapter_version=None,
            confidence=row["confidence"],
            output=row["output"],
            by_rule=row["by_rule"],
        )
        for document, row in zip(documents, combined, strict=True)
    ]
    scored = score(documents, results, groups)
    metrics = compute_metrics(scored, [])
    # The canary documents are not sent again. A canary name can only reach the output of a pair
    # if an adapter produces it, so the pair gets the worse of the two.
    leaks = [m["metrics"]["canary_leak_rate"] for m in (first, second)]
    values = [leak["value"] for leak in leaks if leak["value"] is not None]
    metrics["canary_leak_rate"] = {
        "value": max(values) if values else None,
        "documents": leaks[0]["documents"],
    }
    commit, dirty = git_state()
    wall = sum(m["performance"]["wall_seconds"] for m in (first, second))
    manifest = first["manifest"]
    return {
        "platform": first["platform"],
        "system": name,
        "split": first["split"],
        "manifest": {
            "configuration": {
                "members": [m["system"] for m in (first, second)],
                "rule": "automated only if both answers have the same items, reverse-charge "
                "flag and total, and neither adapter sent the document to review",
                "confidence": confidence,
            },
            "members": [
                {
                    key: m["manifest"].get(key)
                    for key in (
                        "model",
                        "model_revision",
                        "adapter_hash",
                        "git_commit",
                        "threshold",
                    )
                }
                for m in (first, second)
            ],
            "vllm_version": manifest.get("vllm_version"),
            "gpu": manifest.get("gpu"),
            "few_shot": False,
            "data_round": manifest["data_round"],
            "data_generator": manifest.get("data_generator"),
            "data_sha256": manifest["data_sha256"],
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": SEED,
            "documents": len(documents),
        },
        "confidence_threshold": {
            "value": None,
            "source": "each adapter keeps its own threshold"
            if confidence == "each"
            else "ignored, only the agreement of the adapters counts",
        },
        # Both adapters answer every document on one GPU, so the times add up. The latency of a
        # pair is not worked out from the two runs.
        "performance": {
            "wall_seconds": wall,
            "documents_per_second": len(documents) / wall if wall else None,
            "latency_p50_seconds": None,
            "latency_p95_seconds": None,
            "basis": "sum of the wall times of the two members",
        },
        "metrics": metrics,
        "slices": compute_slices(scored),
        "documents": document_rows(scored),
    }


def _load(out_dir: Path, platform: str, system: str, split: str) -> dict[str, Any] | None:
    path = results_file(out_dir, platform, system, split)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine two systems into a pair configuration.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--first", required=True, help="first system")
    parser.add_argument("--second", required=True, help="second system")
    parser.add_argument("--name", required=True, help="name of the pair in the results folder")
    parser.add_argument(
        "--confidence",
        choices=("each", "ignore"),
        default="each",
        help="each: keep the confidence threshold of every adapter; ignore: only agreement counts",
    )
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, help="default: all they share")
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    parser.add_argument("--out-dir", type=Path, help="default: results/round<N>/eval")
    args = parser.parse_args(argv)
    if not SYSTEM_NAME.match(args.name):
        sys.exit(f"error: name {args.name!r} has characters that are not allowed")
    try:
        out_dir = args.out_dir or eval_dir(args.round)
        data_round = resolve_round(args.round, DATA_DIR)
    except ValueError as error:
        sys.exit(f"error: {error}")

    platform = load_platform(args.platform)
    groups = account_groups(platform.rules)
    written = 0
    for split in args.splits or ALL_SPLITS:
        first = _load(out_dir, args.platform, args.first, split)
        second = _load(out_dir, args.platform, args.second, split)
        if first is None or second is None:
            print(f"{split}: skipped, no results for both systems")
            continue
        try:
            documents = load_documents(args.platform, split, DATA_DIR, data_round)
            combined = combine_results(first, second, args.name, documents, groups, args.confidence)
        except ValueError as error:
            sys.exit(f"error: {split}: {error}")
        path = results_file(out_dir, args.platform, args.name, split)
        write_results(combined, path)
        rows = combined["documents"]
        automated = sum(not row["needs_review"] for row in rows)
        print(f"{split}: {automated} of {len(rows)} documents automated -> {path}")
        written += 1
    if not written:
        sys.exit("error: no split has results for both systems")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score saved results again with the pipeline checks that were added after the run.

Run with `make rescore SYSTEMS="..." ROUND=2`. A saved row holds the answer of the model, so the
checks that need no model can be applied to it again: `heavy_noise` (a document check, which
would have stopped the model from being called) and `vat_mismatch` (a check on the answer). No
model runs. The results go to `results/round<N>/rescored/`, in the layout of `eval/`, and the
files in `eval/` stay as they were run. The confidence thresholds are the ones of the run.

For each system and split the run prints what the checks changed: documents they sent to review
that had a wrong answer (caught) and documents they sent to review that had a right answer
(false flags).
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adapt.config import Platform, load_platform
from adapt.data.rounds import rescored_dir, resolve_round
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
from adapt.eval.scoring import (
    EvalDocument,
    compute_metrics,
    compute_slices,
    document_rows,
    result_from_row,
    score,
)
from adapt.pipeline import document_problems
from adapt.vat_check import vat_mismatch

CHECKS = ("heavy_noise", "vat_mismatch")
CONFIDENCE_REASONS = ("no_confidence", "low_confidence")


def rescore_row(
    row: Mapping[str, Any], document: EvalDocument, platform: Platform
) -> dict[str, Any]:
    """The row as the pipeline would have produced it with the checks of today."""
    problems = document_problems(document.text, document.chart, platform)
    if problems:
        # A document check stops the pipeline before the model is called.
        return {
            **row,
            "needs_review": True,
            "reasons": list(problems),
            "output": None,
            "confidence": None,
            "item_confidences": None,
            "by_rule": True,
        }
    output = row["output"]
    if (
        output
        and output["items"]
        and vat_mismatch(document.text, output["items"], bool(output.get("reverse_charge")))
    ):
        reasons = [r for r in row["reasons"] if r not in CONFIDENCE_REASONS]
        reasons.append("vat_mismatch")
        reasons += [r for r in row["reasons"] if r in CONFIDENCE_REASONS]
        return {
            **row,
            "needs_review": True,
            "reasons": reasons,
            "by_rule": "model_flagged" not in reasons,
        }
    return dict(row)


def changes(old: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]) -> Counter[str]:
    """What the checks did to the documents that were automated: caught or falsely flagged."""
    count: Counter[str] = Counter()
    for before, after in zip(old, new, strict=True):
        if before["needs_review"] or not after["needs_review"]:
            continue
        check = next(r for r in after["reasons"] if r not in before["reasons"])
        wrong = before["expected_review"] or not before["answer_correct"]
        count[f"{check} {'caught' if wrong else 'false'}"] += 1
    return count


def rescore(
    results: Mapping[str, Any],
    documents: Sequence[EvalDocument],
    platform: Platform,
) -> tuple[dict[str, Any], Counter[str]]:
    """The saved results of one split scored again, and what the checks changed."""
    rows = results["documents"]
    if [d.id for d in documents] != [row["id"] for row in rows]:
        raise ValueError("the documents do not match the results")
    if any("output" not in row or "by_rule" not in row for row in rows):
        raise ValueError(
            f"{results['system']} was run before the results kept the model output, "
            "so it cannot be scored again"
        )
    new_rows = [rescore_row(row, d, platform) for row, d in zip(rows, documents, strict=True)]
    scored = score(
        documents,
        [result_from_row(row, results["system"]) for row in new_rows],
        account_groups(platform.rules),
    )
    metrics = compute_metrics(scored, [])
    metrics["canary_leak_rate"] = results["metrics"]["canary_leak_rate"]
    commit, dirty = git_state()
    rescored = {
        **results,
        "manifest": {
            **results["manifest"],
            "rescored": {"checks": list(CHECKS), "git_commit": commit, "git_dirty": dirty},
        },
        "metrics": metrics,
        "slices": compute_slices(scored),
        "documents": document_rows(scored),
    }
    # The rows keep the document check outcome of today, so the old rows are compared by value.
    return rescored, changes(rows, rescored["documents"])


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score saved results again with newer checks.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--systems", nargs="+", required=True, help="systems to score again")
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, help="default: all they have")
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    parser.add_argument("--in-dir", type=Path, help="default: results/round<N>/eval")
    parser.add_argument("--out-dir", type=Path, help="default: results/round<N>/rescored")
    args = parser.parse_args(argv)
    for system in args.systems:
        if not SYSTEM_NAME.match(system):
            sys.exit(f"error: system name {system!r} has characters that are not allowed")
    try:
        data_round = resolve_round(args.round, DATA_DIR)
        in_dir = args.in_dir or Path("results") / f"round{data_round}" / "eval"
        out_dir = args.out_dir or rescored_dir(data_round)
    except ValueError as error:
        sys.exit(f"error: {error}")
    platform = load_platform(args.platform)

    written = 0
    for system in args.systems:
        for split in args.splits or ALL_SPLITS:
            results = _load(results_file(in_dir, args.platform, system, split))
            if results is None:
                continue
            documents = load_documents(args.platform, split, DATA_DIR, data_round)
            try:
                rescored, counted = rescore(results, documents, platform)
            except ValueError as error:
                sys.exit(f"error: {system} {split}: {error}")
            write_results(rescored, results_file(out_dir, args.platform, system, split))
            written += 1
            shown = ", ".join(f"{key} {value}" for key, value in sorted(counted.items()))
            print(f"{system} {split}: {shown or 'no change'}")
    if not written:
        sys.exit("error: no results found")
    print(f"{written} results files written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

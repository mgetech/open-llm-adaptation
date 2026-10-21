"""Score saved results again with the pipeline checks that were added after the run.

Run with `make rescore SYSTEMS="..." ROUND=2`. A saved row holds the answer of the model, so the
checks that need no model can be applied to it again: `heavy_noise` (a document check, which
would have stopped the model from being called) and the checks on the answer in
`adapt.answer_checks` (`vat_mismatch`, `mixed_private`, `reverse_charge_cue`). No model runs. The
results go to `results/round<N>/rescored/`, in the layout of `eval/`, and the files in `eval/`
stay as they were run. The confidence thresholds are the ones of the run, unless `RETUNE=1` is
given: then the threshold is chosen again on the rescored `dev` rows (accuracy target and limit
on critical errors, see `adapt.eval.threshold`) and used as it is on every other split.

For each system and split the run prints what the checks changed: documents they sent to review
that had a wrong answer (caught) and documents they sent to review that had a right answer
(false flags).

It also prints the effect of each of the two newest rules alone and together, on `dev`,
`dev_real` and `test_ood` only: automation, review recall, critical errors and accuracy on the
automated documents, next to the numbers without them. The gold sets are not part of that table.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adapt.answer_checks import ANSWER_RULES, answer_flags, private_accounts
from adapt.config import Platform, load_platform
from adapt.data.rounds import rescored_dir, resolve_round
from adapt.eval.metrics import (
    account_groups,
    accuracy_on_automated,
    automation_rate,
    critical_error_rate,
    review_recall,
)
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
from adapt.eval.threshold import Tuned, apply_threshold, tune_threshold, tuning_record
from adapt.pipeline import document_problems

CHECKS = ("heavy_noise", *ANSWER_RULES)
CONFIDENCE_REASONS = ("no_confidence", "low_confidence")
# The rules whose effect is shown one by one, and the splits they are shown on. The gold sets are
# left out on purpose.
NEW_RULES = ("mixed_private", "reverse_charge_cue")
EFFECT_SPLITS = ("dev", "dev_real", "test_ood")


def rescore_row(
    row: Mapping[str, Any],
    document: EvalDocument,
    platform: Platform,
    rules: Sequence[str] = ANSWER_RULES,
) -> dict[str, Any]:
    """The row as the pipeline would have produced it with the checks of today.

    `rules` are the checks on the answer to apply. The default is all of them."""
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
    if output and output["items"]:
        private = private_accounts(platform.rules.accounts)
        found = [
            rule
            for rule in answer_flags(document.text, output, private, rules)
            if rule not in row["reasons"]
        ]
        if found:
            reasons = [r for r in row["reasons"] if r not in CONFIDENCE_REASONS]
            reasons += found
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


def rows_without_limit(
    results: Mapping[str, Any], documents: Sequence[EvalDocument], platform: Platform
) -> list[dict[str, Any]]:
    """The rows of a split with the checks of today and no confidence limit. A document that
    only the confidence limit sent to review has the severity of its answer again, so any
    threshold can be tried on these rows."""
    new_rows = [
        rescore_row(row, d, platform)
        for row, d in zip(results["documents"], documents, strict=True)
    ]
    unlimited = [apply_threshold(result_from_row(row, results["system"]), 0.0) for row in new_rows]
    return document_rows(score(documents, unlimited, account_groups(platform.rules)))


def tune_on_dev(
    results: Mapping[str, Any], documents: Sequence[EvalDocument], platform: Platform
) -> Tuned:
    """The threshold chosen again on saved dev results, with the gate's targets."""
    gate = platform.config.gate
    return tune_threshold(
        rows_without_limit(results, documents, platform),
        gate.accuracy_automated_min,
        gate.critical_error_rate_max,
    )


def rescore(
    results: Mapping[str, Any],
    documents: Sequence[EvalDocument],
    platform: Platform,
    tuned: Tuned | None = None,
    split: str | None = None,
) -> tuple[dict[str, Any], Counter[str]]:
    """The saved results of one split scored again, and what the checks changed. With `tuned`,
    the threshold of those dev results is used instead of the one of the run."""
    rows = results["documents"]
    if [d.id for d in documents] != [row["id"] for row in rows]:
        raise ValueError("the documents do not match the results")
    if any("output" not in row or "by_rule" not in row for row in rows):
        raise ValueError(
            f"{results['system']} was run before the results kept the model output, "
            "so it cannot be scored again"
        )
    new_rows = [rescore_row(row, d, platform) for row, d in zip(rows, documents, strict=True)]
    new_results = [result_from_row(row, results["system"]) for row in new_rows]
    threshold = None
    if tuned is not None:
        threshold = tuned.threshold if tuned.threshold is not None else 0.0
        new_results = [apply_threshold(result, threshold) for result in new_results]
    scored = score(documents, new_results, account_groups(platform.rules))
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
    if tuned is not None:
        rescored["manifest"]["threshold"] = threshold
        rescored["confidence_threshold"] = {"value": threshold, "source": "tuned again on dev"} | (
            tuning_record(tuned) if split == "dev" else {}
        )
    # The rows keep the document check outcome of today, so the old rows are compared by value.
    return rescored, changes(rows, rescored["documents"])


def rule_effects(
    results: Mapping[str, Any], documents: Sequence[EvalDocument], platform: Platform
) -> list[dict[str, Any]]:
    """The numbers of one split without the newest rules, with each alone, and with both.

    The first line is the base: the checks that were there before, so each later line shows what
    its rule adds. `caught` counts the documents the rule sent to review that had a wrong answer
    or should have gone to review, and `false` those that had a right answer."""
    groups = account_groups(platform.rules)
    base_rules = tuple(rule for rule in ANSWER_RULES if rule not in NEW_RULES)
    variants = [("without them", ()), *[(rule, (rule,)) for rule in NEW_RULES]]
    variants.append(("both", NEW_RULES))
    rows = results["documents"]
    base_rows: list[dict[str, Any]] = []
    lines = []
    for label, added in variants:
        new_rows = [
            rescore_row(row, d, platform, (*base_rules, *added))
            for row, d in zip(rows, documents, strict=True)
        ]
        if not added:
            base_rows = new_rows
        scored = score(
            documents, [result_from_row(row, results["system"]) for row in new_rows], groups
        )
        cases = [s.case for s in scored]
        counted = changes(base_rows, new_rows)
        lines.append(
            {
                "label": label,
                "automation_rate": automation_rate(cases),
                "review_recall": review_recall(cases),
                "critical_error_rate": critical_error_rate(cases, groups),
                "accuracy_on_automated": accuracy_on_automated(cases),
                "caught": sum(v for k, v in counted.items() if k.endswith(" caught")),
                "false": sum(v for k, v in counted.items() if k.endswith(" false")),
            }
        )
    return lines


def format_effects(system: str, split: str, count: int, lines: Sequence[Mapping[str, Any]]) -> str:
    """The effect table of one system and split, as text."""

    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    columns = "automation  review recall  critical errors  accuracy on automated  caught  false"
    out = [f"{system} {split} ({count} documents)", f"  {'':20}{columns}"]
    for line in lines:
        out.append(
            f"  {line['label']:20}"
            f"{number(line['automation_rate']):>10}  {number(line['review_recall']):>13}"
            f"  {number(line['critical_error_rate']):>15}  "
            f"{number(line['accuracy_on_automated']):>21}  {line['caught']:>6}  {line['false']:>5}"
        )
    return "\n".join(out)


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
    parser.add_argument(
        "--retune",
        action="store_true",
        help="choose the confidence threshold again on dev and use it on every split",
    )
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
    effects: list[str] = []
    effect_splits: set[str] = set()
    for system in args.systems:
        tuned = None
        if args.retune:
            dev = _load(results_file(in_dir, args.platform, system, "dev"))
            if dev is None:
                sys.exit(f"error: {system}: --retune needs the saved dev results")
            try:
                dev_documents = load_documents(args.platform, "dev", DATA_DIR, data_round)
                tuned = tune_on_dev(dev, dev_documents, platform)
            except ValueError as error:
                sys.exit(f"error: {system} dev: {error}")
            print(f"{system}: threshold chosen again on dev: {tuned.threshold}")
        for split in args.splits or ALL_SPLITS:
            results = _load(results_file(in_dir, args.platform, system, split))
            if results is None:
                continue
            documents = load_documents(args.platform, split, DATA_DIR, data_round)
            try:
                rescored, counted = rescore(results, documents, platform, tuned, split)
            except ValueError as error:
                sys.exit(f"error: {system} {split}: {error}")
            write_results(rescored, results_file(out_dir, args.platform, system, split))
            written += 1
            shown = ", ".join(f"{key} {value}" for key, value in sorted(counted.items()))
            print(f"{system} {split}: {shown or 'no change'}")
            if split in EFFECT_SPLITS:
                effect_splits.add(split)
                lines = rule_effects(results, documents, platform)
                effects.append(format_effects(system, split, len(documents), lines))
    if not written:
        sys.exit("error: no results found")
    print(f"{written} results files written to {out_dir}")
    if effects:
        print("\nEffect of the newest rules (the gold sets are not shown):\n")
        print("\n\n".join(effects))
    wanted = [split for split in EFFECT_SPLITS if not args.splits or split in args.splits]
    missing = [split for split in wanted if split not in effect_splits]
    if missing:
        print(f"\nNo saved results for: {', '.join(missing)}. Run them to see the effect there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

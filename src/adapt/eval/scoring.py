"""Score each document, then work out the metrics, the slices and the per-document rows."""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from adapt.eval.metrics import (
    INJECTION_CATEGORY,
    WEIGHTS,
    Case,
    Severity,
    accuracy_on_automated,
    answer_correct,
    automation_rate,
    canary_leak_rate,
    document_accuracy,
    document_correct,
    error_severity,
    item_scores,
)
from adapt.eval.stats import bootstrap_ratio
from adapt.pipeline import Result


@dataclass(frozen=True)
class EvalDocument:
    id: str
    text: str
    chart: str
    target: Mapping[str, Any]
    expected_review: bool
    category: str | None = None
    layout: str | None = None
    noise: str | None = None
    seen_supplier: bool | None = None


def to_case(document: EvalDocument, result: Result) -> Case:
    output = result.output or {}
    return Case(
        target=document.target,
        expected_review=document.expected_review,
        needs_review=result.needs_review,
        items=result.items,
        reverse_charge=output.get("reverse_charge", False),
        supplier=output.get("supplier", ""),
        category=document.category,
    )


@dataclass(frozen=True)
class Scored:
    """One document with everything the metrics need, worked out once."""

    document: EvalDocument
    result: Result
    case: Case
    correct: bool
    answer_correct: bool
    severity: Severity | None
    true_positives: int
    false_positives: int
    false_negatives: int


def score(
    documents: Sequence[EvalDocument], results: Sequence[Result], groups: Mapping[str, str]
) -> list[Scored]:
    scored = []
    for document, result in zip(documents, results, strict=True):
        case = to_case(document, result)
        items = item_scores([case])
        scored.append(
            Scored(
                document,
                result,
                case,
                document_correct(case),
                answer_correct(case),
                error_severity(case, groups),
                items.true_positives,
                items.false_positives,
                items.false_negatives,
            )
        )
    return scored


# Each headline number is the sum of the numerators over the sum of the denominators, with one
# (numerator, denominator) pair per document. That form gives a bootstrap interval for all of them.
TERMS: dict[str, Callable[[Scored], tuple[float, float]]] = {
    "document_accuracy": lambda s: (s.correct, 1),
    "automation_rate": lambda s: (
        s.case.automated and not s.case.expected_review,
        not s.case.expected_review,
    ),
    "accuracy_on_automated": lambda s: (s.case.automated and s.correct, s.case.automated),
    "critical_error_rate": lambda s: (s.severity == "critical", s.case.automated),
    "weighted_error_cost": lambda s: (WEIGHTS[s.severity] * 1000 if s.severity else 0, 1),
    "review_recall": lambda s: (
        s.case.needs_review and s.case.expected_review,
        s.case.expected_review,
    ),
    "injection_success_rate": lambda s: (
        s.case.category == INJECTION_CATEGORY and not s.correct,
        s.case.category == INJECTION_CATEGORY,
    ),
    "item_precision": lambda s: (s.true_positives, s.true_positives + s.false_positives),
    "item_recall": lambda s: (s.true_positives, s.true_positives + s.false_negatives),
    "item_f1": lambda s: (
        2 * s.true_positives,
        2 * s.true_positives + s.false_positives + s.false_negatives,
    ),
    "invalid_json_rate": lambda s: ("invalid_output" in s.result.reasons, 1),
    "invented_account_rate": lambda s: ("unknown_account" in s.result.reasons, 1),
}


def compute_metrics(
    scored: Sequence[Scored], canaries: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """Every headline number with a 95% bootstrap interval, plus the severity counts."""
    metrics: dict[str, Any] = {}
    for name, terms in TERMS.items():
        pairs = [terms(s) for s in scored]
        interval = bootstrap_ratio([n for n, _ in pairs], [d for _, d in pairs])
        metrics[name] = (
            {"value": None, "low": None, "high": None}
            if interval is None
            else {"value": interval.estimate, "low": interval.low, "high": interval.high}
        )
    metrics["canary_leak_rate"] = {
        "value": canary_leak_rate(canaries),
        "documents": len(canaries),
    }
    counts: dict[str, int] = {"critical": 0, "major": 0, "minor": 0}
    for s in scored:
        if s.severity:
            counts[s.severity] += 1
    metrics["severity_counts"] = counts
    return metrics


def _bucket(count: int) -> str:
    return str(count) if count < 4 else "4+"


def _vat_rates(document: EvalDocument) -> str:
    rates = {item["vat_rate"] for item in document.target["items"]}
    return "none" if not rates else "single" if len(rates) == 1 else "mixed"


def _supplier(document: EvalDocument) -> str | None:
    if document.seen_supplier is None:
        return None
    return "seen" if document.seen_supplier else "unseen"


SLICES: dict[str, Callable[[EvalDocument], str | None]] = {
    "items": lambda d: _bucket(len(d.target["items"])),
    "vat_rates": _vat_rates,
    "layout": lambda d: d.layout,
    "noise": lambda d: d.noise,
    "supplier": _supplier,
    "category": lambda d: d.category or "none",
}


def compute_slices(scored: Sequence[Scored]) -> dict[str, dict[str, dict[str, Any]]]:
    """Document accuracy, automation rate and accuracy on automated documents per slice."""
    slices: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, key in SLICES.items():
        members: dict[str, list[Case]] = defaultdict(list)
        for s in scored:
            value = key(s.document)
            if value is not None:
                members[value].append(s.case)
        if members:
            slices[dimension] = {
                value: {
                    "n": len(cases),
                    "document_accuracy": document_accuracy(cases),
                    "automation_rate": automation_rate(cases),
                    "accuracy_on_automated": accuracy_on_automated(cases),
                }
                for value, cases in sorted(members.items())
            }
    return slices


def document_rows(scored: Sequence[Scored]) -> list[dict[str, Any]]:
    """One row per document, enough to recompute rates, compare systems and tune the threshold.

    `output` is what the model answered (None if it gave no valid answer) and
    `item_confidences` the probability of each item's account, so that the errors can be read.
    `by_rule` is true when a rule flagged the document: a document check (the model was not
    called), or the printed-VAT check on an answer the model did not flag itself.
    """
    return [
        {
            "id": s.document.id,
            "expected_review": s.document.expected_review,
            "needs_review": s.result.needs_review,
            "correct": s.correct,
            "answer_correct": s.answer_correct,
            "severity": s.severity,
            "reasons": list(s.result.reasons),
            "confidence": s.result.confidence,
            "item_confidences": s.result.item_confidences,
            "output": s.result.output,
            "by_rule": s.result.by_rule,
        }
        for s in scored
    ]

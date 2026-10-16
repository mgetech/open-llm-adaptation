"""Tune the confidence threshold on the dev split.

A document is automated at threshold `t` when nothing else sent it to review and its
confidence is at least `t`. The rows of a run hold what is needed to try any `t` without
running the model again. The threshold chosen is the lowest one, so the most documents are
automated, at which the accuracy on automated documents reaches the target.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from adapt.pipeline import Result

LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    automated: int
    automation_rate: float | None
    accuracy_on_automated: float | None


@dataclass(frozen=True)
class Tuned:
    """The curve and the point chosen from it. `chosen` is None if no point reaches the target."""

    target_accuracy: float
    curve: list[CurvePoint]
    chosen: CurvePoint | None

    @property
    def threshold(self) -> float | None:
        return self.chosen.threshold if self.chosen else None


def apply_threshold(result: Result, threshold: float) -> Result:
    """Set the low-confidence outcome of `result` for `threshold`, as the pipeline would."""
    reasons = [reason for reason in result.reasons if reason != LOW_CONFIDENCE]
    if result.confidence is not None and result.confidence < threshold:
        reasons.append(LOW_CONFIDENCE)
    return replace(result, reasons=reasons, needs_review=bool(reasons))


def _automated(row: Mapping[str, Any], threshold: float) -> bool:
    if any(reason != LOW_CONFIDENCE for reason in row["reasons"]):
        return False
    confidence = row["confidence"]
    return confidence is None or confidence >= threshold


def _point(rows: Sequence[Mapping[str, Any]], threshold: float) -> CurvePoint:
    automated = [row for row in rows if _automated(row, threshold)]
    automatable = [row for row in rows if not row["expected_review"]]
    # An automated document is only right if it should not have been reviewed.
    right = sum(row["answer_correct"] and not row["expected_review"] for row in automated)
    done = sum(not row["expected_review"] for row in automated)
    return CurvePoint(
        threshold,
        len(automated),
        done / len(automatable) if automatable else None,
        right / len(automated) if automated else None,
    )


def coverage_curve(rows: Sequence[Mapping[str, Any]]) -> list[CurvePoint]:
    """Automation rate and accuracy on automated documents at each threshold worth trying.

    The thresholds are 0 and every confidence of a document that nothing else sent to review.
    """
    candidates = {0.0}
    for row in rows:
        if row["confidence"] is not None and _automated(row, 0.0):
            candidates.add(row["confidence"])
    return [_point(rows, threshold) for threshold in sorted(candidates)]


def tune_threshold(rows: Sequence[Mapping[str, Any]], target_accuracy: float) -> Tuned:
    """The lowest threshold whose accuracy on automated documents is at least the target."""
    curve = coverage_curve(rows)
    for point in curve:
        accuracy = point.accuracy_on_automated
        if accuracy is not None and accuracy >= target_accuracy:
            return Tuned(target_accuracy, curve, point)
    return Tuned(target_accuracy, curve, None)


def tuning_record(tuned: Tuned) -> dict[str, Any]:
    """The part of the results file that stores the tuning."""
    chosen = tuned.chosen
    return {
        "value": tuned.threshold,
        "target_accuracy": tuned.target_accuracy,
        "target_reached": chosen is not None,
        "automation_rate": chosen.automation_rate if chosen else None,
        "accuracy_on_automated": chosen.accuracy_on_automated if chosen else None,
        "curve": [asdict(point) for point in tuned.curve],
    }


def read_threshold(path: Path) -> float | None:
    """The threshold stored in a dev results file. None if the target was not reached."""
    results = json.loads(path.read_text(encoding="utf-8"))
    value = results["confidence_threshold"]["value"]
    return None if value is None else float(value)

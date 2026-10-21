import json

import pytest

from adapt.client import ModelClient
from adapt.config import load_platform
from adapt.eval.runner import load_documents
from adapt.eval.threshold import (
    apply_threshold,
    coverage_curve,
    read_threshold,
    tune_threshold,
    tuning_record,
)
from adapt.pipeline import Pipeline, Result
from fake_model import Model


def row(confidence, right, *, expected_review=False, reasons=(), severity=None):
    """One dev document. `right` says whether the answer matches the expected output."""
    return {
        "expected_review": expected_review,
        "reasons": list(reasons),
        "confidence": confidence,
        "answer_correct": right,
        "severity": severity,
    }


# Ten automatable documents. The confidences and the right and wrong answers are chosen so the
# accuracy at each threshold can be counted by hand:
#   threshold  documents at or above  right  accuracy
#   0.4        10                     5      0.500
#   0.5        9                      5      0.556
#   0.6        8                      5      0.625
#   0.7        7                      5      0.714
#   0.8        6                      5      0.833
#   0.85       5                      4      0.800
#   0.9        4                      4      1.000
#   0.95       3                      3      1.000
#   0.97       2                      2      1.000
#   0.99       1                      1      1.000
SYNTHETIC = [
    row(0.99, True),
    row(0.97, True),
    row(0.95, True),
    row(0.90, True),
    row(0.85, False),
    row(0.80, True),
    row(0.70, False),
    row(0.60, False),
    row(0.50, False),
    row(0.40, False),
]


def accuracy_at(rows, threshold):
    points = {p.threshold: p for p in coverage_curve(rows)}
    return points[threshold].accuracy_on_automated


def test_the_curve_matches_the_hand_counted_accuracies():
    expected = {
        0.4: 5 / 10,
        0.5: 5 / 9,
        0.6: 5 / 8,
        0.7: 5 / 7,
        0.8: 5 / 6,
        0.85: 4 / 5,
        0.9: 4 / 4,
        0.95: 3 / 3,
        0.97: 2 / 2,
        0.99: 1 / 1,
    }
    for threshold, accuracy in expected.items():
        assert accuracy_at(SYNTHETIC, threshold) == pytest.approx(accuracy), threshold


def test_the_curve_starts_at_zero_and_covers_every_confidence_once():
    curve = coverage_curve(SYNTHETIC)
    assert [p.threshold for p in curve] == [
        0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99
    ]  # fmt: skip
    assert curve[0].automated == 10 and curve[0].automation_rate == 1.0
    assert curve[-1].automated == 1 and curve[-1].automation_rate == pytest.approx(0.1)


def test_the_tuned_threshold_is_the_lowest_one_that_reaches_the_target():
    assert tune_threshold(SYNTHETIC, 0.95).threshold == 0.9
    assert tune_threshold(SYNTHETIC, 0.95).chosen.automated == 4
    assert tune_threshold(SYNTHETIC, 1.0).threshold == 0.9


def test_a_lower_target_gives_a_lower_threshold():
    assert tune_threshold(SYNTHETIC, 0.8).threshold == 0.8
    assert tune_threshold(SYNTHETIC, 0.7).threshold == 0.7
    assert tune_threshold(SYNTHETIC, 0.5).threshold == 0.0


def test_the_lowest_threshold_wins_even_when_accuracy_dips_above_it():
    # At 0.85 the accuracy is 0.8, below the 0.833 at 0.8, so the curve is not monotone.
    assert accuracy_at(SYNTHETIC, 0.85) < accuracy_at(SYNTHETIC, 0.8)
    assert tune_threshold(SYNTHETIC, 0.8).threshold == 0.8


def test_a_target_that_cannot_be_reached_gives_no_threshold():
    rows = [row(0.9, False), row(0.8, False), row(0.7, True)]
    tuned = tune_threshold(rows, 0.95)
    assert tuned.threshold is None and tuned.chosen is None
    record = tuning_record(tuned)
    assert record["value"] is None and record["target_reached"] is False
    assert record["automation_rate"] is None and record["accuracy_on_automated"] is None
    assert len(record["curve"]) == 4


def test_documents_sent_to_review_for_another_reason_are_never_automated():
    other = [
        row(0.99, False, reasons=["total_mismatch"]),
        row(None, False, reasons=["no_confidence"]),
    ]
    rows = [*SYNTHETIC, *other]
    assert [p.threshold for p in coverage_curve(rows)] == [
        p.threshold for p in coverage_curve(SYNTHETIC)
    ]
    point = coverage_curve(rows)[0]
    assert point.automated == 10
    assert point.automation_rate == pytest.approx(10 / 12)
    assert tune_threshold(rows, 0.95).threshold == 0.9


def test_a_document_that_should_be_reviewed_but_is_automated_counts_as_wrong():
    rows = [*SYNTHETIC, row(0.99, True, expected_review=True)]
    assert accuracy_at(rows, 0.9) == pytest.approx(4 / 5)
    assert accuracy_at(rows, 0.99) == pytest.approx(1 / 2)
    # It does not count as an automatable document, so the automation rate stays the same.
    assert coverage_curve(rows)[0].automation_rate == 1.0
    # Whatever the threshold, the wrongly automated document keeps the accuracy below 0.95.
    assert tune_threshold(rows, 0.95).threshold is None


def test_a_document_that_was_sent_to_review_for_low_confidence_is_tried_again():
    rows = [row(0.99, True), row(0.3, False, reasons=["low_confidence"]), row(0.9, True)]
    curve = coverage_curve(rows)
    assert [p.threshold for p in curve] == [0.0, 0.3, 0.9, 0.99]
    assert curve[0].automated == 3
    assert curve[1].automated == 3
    assert curve[2].automated == 2


def test_the_threshold_decides_with_at_least_not_more_than():
    assert coverage_curve([row(0.5, True)])[1].automated == 1


def test_without_documents_there_is_nothing_to_tune():
    curve = coverage_curve([])
    assert len(curve) == 1 and curve[0].automated == 0
    assert curve[0].automation_rate is None and curve[0].accuracy_on_automated is None
    assert tune_threshold([], 0.95).threshold is None


def test_the_tuning_record_can_be_written_as_json():
    record = tuning_record(tune_threshold(SYNTHETIC, 0.95))
    assert json.loads(json.dumps(record)) == record
    assert record["value"] == 0.9 and record["target_reached"] is True
    assert record["automation_rate"] == pytest.approx(0.4)
    assert record["accuracy_on_automated"] == 1.0
    assert record["curve"][0] == {
        "threshold": 0.0,
        "automated": 10,
        "automation_rate": 1.0,
        "accuracy_on_automated": 0.5,
        "critical_errors": 0,
        "critical_error_rate": 0.0,
    }


def test_a_stored_threshold_is_read_back(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps({"confidence_threshold": {"value": 0.83}}), encoding="utf-8")
    assert read_threshold(path) == 0.83
    path.write_text(json.dumps({"confidence_threshold": {"value": None}}), encoding="utf-8")
    assert read_threshold(path) is None


def result(confidence, reasons=()):
    return Result(
        request_id="r",
        needs_review=bool(reasons),
        reasons=list(reasons),
        items=[],
        booking_lines=[],
        adapter="a",
        adapter_version=None,
        confidence=confidence,
    )


def test_a_low_confidence_adds_the_reason_and_sends_the_document_to_review():
    low = apply_threshold(result(0.4), 0.5)
    assert low.needs_review and low.reasons == ["low_confidence"]
    assert not apply_threshold(result(0.5), 0.5).needs_review
    assert not apply_threshold(result(0.6), 0.5).needs_review


def test_applying_a_lower_threshold_takes_the_reason_away_again():
    flagged = apply_threshold(result(0.4), 0.5)
    freed = apply_threshold(flagged, 0.3)
    assert freed.reasons == [] and not freed.needs_review


def test_other_reasons_stay_and_the_reason_is_not_added_twice():
    flagged = result(0.4, ["total_mismatch"])
    once = apply_threshold(flagged, 0.5)
    twice = apply_threshold(once, 0.5)
    assert once.reasons == ["total_mismatch", "low_confidence"]
    assert twice.reasons == once.reasons
    assert apply_threshold(flagged, 0.1).reasons == ["total_mismatch"]


def test_a_result_without_a_confidence_is_left_alone():
    flagged = result(None, ["model_flagged"])
    assert apply_threshold(flagged, 0.99) == flagged


def test_applying_a_threshold_afterwards_gives_what_the_pipeline_gives_directly(server):
    """The tuning applies the threshold to results from a run without one."""
    platform = load_platform("tax-booking")
    documents = load_documents("tax-booking", "dev")[:120]

    def varied(token):
        return -(int(token) % 7) * 0.2 if token.isdigit() and len(token) == 4 else -0.001

    server.handler = Model(documents, logprob_for=varied)
    with ModelClient(server.url, timeout=5.0) as client:

        def run(threshold):
            pipeline = Pipeline(platform, client, adapter="tax-booking", threshold=threshold)
            return [pipeline.process(d.text, d.chart) for d in documents]

        base = run(0.0)
        confidences = {r.confidence for r in base if r.confidence is not None}
        assert len(confidences) > 3
        for threshold in (0.5, 0.7, 0.9):
            direct = run(threshold)
            for expected, after in zip(direct, base, strict=True):
                applied = apply_threshold(after, threshold)
                assert (applied.needs_review, applied.reasons) == (
                    expected.needs_review,
                    expected.reasons,
                ), threshold
            assert any("low_confidence" in r.reasons for r in direct), threshold


# Six documents. The one at 0.9 has a critical error, the others are right.
WITH_CRITICAL = [
    row(0.99, True),
    row(0.95, True),
    row(0.90, False, severity="critical"),
    row(0.80, True),
    row(0.70, True),
    row(0.60, True),
]


def test_without_a_limit_on_critical_errors_the_accuracy_target_alone_decides():
    tuned = tune_threshold(WITH_CRITICAL, 0.75)
    assert tuned.threshold == 0.0 and tuned.chosen.critical_errors == 1


def test_the_limit_on_critical_errors_raises_the_threshold_above_the_critical_document():
    tuned = tune_threshold(WITH_CRITICAL, 0.75, 0.0)
    assert tuned.threshold == 0.95 and tuned.critical_reached
    assert tuned.chosen.automated == 2 and tuned.chosen.critical_errors == 0
    assert tuned.chosen.critical_error_rate == 0.0


def test_a_critical_error_is_counted_among_automated_documents_only():
    rows = [*WITH_CRITICAL, row(0.5, False, reasons=["vat_mismatch"], severity="critical")]
    assert coverage_curve(rows)[0].critical_errors == 1


def test_one_critical_error_is_within_a_limit_of_one_in_two_hundred_automated_documents():
    rows = [row(0.99, True) for _ in range(199)] + [row(0.98, False, severity="critical")]
    assert tune_threshold(rows, 0.95, 0.005).threshold == 0.0
    assert tune_threshold(rows[1:], 0.95, 0.005).threshold == 0.99


def test_when_no_threshold_meets_the_critical_limit_the_accuracy_target_alone_is_used():
    rows = [row(0.99, False, severity="critical"), row(0.9, True), row(0.8, True)]
    tuned = tune_threshold(rows, 0.6, 0.005)
    assert tuned.threshold == 0.0 and not tuned.critical_reached
    record = tuning_record(tuned)
    assert record["target_reached"] is True and record["critical_target_reached"] is False
    assert record["target_critical_error_rate"] == 0.005
    assert record["critical_error_rate"] == pytest.approx(1 / 3)


def test_the_record_says_that_both_targets_were_reached():
    record = tuning_record(tune_threshold(WITH_CRITICAL, 0.75, 0.0))
    assert record["target_reached"] and record["critical_target_reached"]
    assert record["critical_error_rate"] == 0.0 and json.loads(json.dumps(record)) == record


def test_rows_without_a_severity_have_no_critical_errors():
    plain = [{k: v for k, v in r.items() if k != "severity"} for r in SYNTHETIC]
    assert coverage_curve(plain)[0].critical_errors == 0

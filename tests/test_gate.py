import copy

import pytest

from adapt.config import load_platform
from adapt.eval.gate import (
    GATE_SPLITS,
    Check,
    check_gate,
    format_table,
    main,
    report_only,
    rule_report,
)
from adapt.eval.runner import (
    RunSettings,
    evaluate,
    load_canaries,
    load_documents,
    results_file,
    write_results,
)
from fake_model import Model, wrong_account_with_low_confidence
from fake_server import FakeServer

GATE = load_platform("tax-booking").config.gate
NAMES = [
    "automation rate",
    "accuracy on automated documents",
    "critical-error rate",
    "fine-tuned vs few-shot, p-value",
    "generalisation gap",
    "review-flag recall",
    "invalid JSON rate",
    "invented account rate",
    "prompt-injection success",
    "canary leak",
]


def rows(count, *, wrong=0, critical=0, reviewed=0, prefix="d"):
    """Documents: the first `wrong` are wrong, the first `critical` have a critical error."""
    return [
        {
            "id": f"{prefix}{i}",
            "needs_review": i >= count - reviewed,
            "correct": i >= wrong,
            "severity": "critical" if i < critical else None,
            "reasons": [],
        }
        for i in range(count)
    ]


def split_results(documents, **values):
    metrics = {name: {"value": value} for name, value in values.items()}
    performance = {
        "documents_per_second": 12.0,
        "latency_p50_seconds": 0.1,
        "latency_p95_seconds": 0.4,
    }
    return {"metrics": metrics, "documents": documents, "performance": performance}


def passing():
    """Results that pass every check."""
    return {
        "gold": split_results(
            rows(20, prefix="g"),
            automation_rate=0.7,
            injection_success_rate=None,
            canary_leak_rate=0.0,
        ),
        "test_iid": split_results(
            rows(20, prefix="i"),
            document_accuracy=0.97,
            injection_success_rate=0.0,
            canary_leak_rate=0.0,
        ),
        "test_ood": split_results(
            rows(20, prefix="o"),
            document_accuracy=0.95,
            injection_success_rate=0.0,
            canary_leak_rate=0.0,
            item_f1=0.9,
            weighted_error_cost=12.0,
        ),
        "edge": split_results(
            rows(20, prefix="e"),
            review_recall=1.0,
            injection_success_rate=0.0,
            canary_leak_rate=0.0,
        ),
    }


def better_than_baseline(results, wins=6):
    """A baseline that is wrong on `wins` test_ood documents where the system is right."""
    baseline = copy.deepcopy(results["test_ood"])
    for row in baseline["documents"][:wins]:
        row["correct"] = False
    return baseline


def check(checks, name):
    return next(c for c in checks if c.name == name)


def failures(checks):
    return [c.name for c in checks if not c.passed]


def test_results_that_meet_every_threshold_pass_all_ten_checks():
    results = passing()
    checks = check_gate(GATE, results, better_than_baseline(results))
    assert [c.name for c in checks] == NAMES
    assert failures(checks) == []


def test_every_threshold_in_the_platform_file_is_used():
    values = {c.limit for c in check_gate(GATE, passing())}
    assert {
        GATE.automation_rate_min,
        GATE.accuracy_automated_min,
        GATE.critical_error_rate_max,
        GATE.mcnemar_p_max,
        GATE.generalisation_gap_max,
        GATE.review_recall_min,
    } <= values


@pytest.mark.parametrize(
    ("value", "passes"), [(0.5, False), (0.59, False), (0.60, True), (0.61, True), (None, False)]
)
def test_automation_rate_must_be_at_least_the_minimum(value, passes):
    results = passing()
    results["gold"]["metrics"]["automation_rate"]["value"] = value
    checks = check_gate(GATE, results, better_than_baseline(results))
    assert (check(checks, "automation rate").passed) is passes
    assert failures(checks) == ([] if passes else ["automation rate"])


@pytest.mark.parametrize(("wrong", "passes"), [(0, True), (2, True), (3, False), (10, False)])
def test_accuracy_on_automated_documents_pools_gold_and_test_ood(wrong, passes):
    # 40 automated documents in all. Two wrong gives 38 / 40 = 0.95, which is enough.
    results = passing()
    results["gold"]["documents"] = rows(20, wrong=wrong, prefix="g")
    checks = check_gate(GATE, results, better_than_baseline(results))
    accuracy = check(checks, "accuracy on automated documents")
    assert accuracy.value == pytest.approx((40 - wrong) / 40)
    assert accuracy.passed is passes


def test_accuracy_counts_automated_documents_only():
    results = passing()
    results["gold"]["documents"] = rows(20, wrong=5, reviewed=5, prefix="g")
    # The 5 wrong documents are the first 5; the last 5 were reviewed and are not counted.
    accuracy = check(check_gate(GATE, results), "accuracy on automated documents")
    assert accuracy.value == pytest.approx((15 - 5 + 20) / (15 + 20))


def test_the_critical_error_rate_is_pooled_and_its_limit_is_inclusive():
    results = passing()
    results["gold"]["documents"] = rows(100, critical=1, prefix="g")
    results["test_ood"]["documents"] = rows(100, prefix="o")
    one_in_200 = check(check_gate(GATE, results), "critical-error rate")
    assert one_in_200.value == pytest.approx(0.005)
    assert one_in_200.passed
    results["gold"]["documents"] = rows(100, critical=2, prefix="g")
    assert not check(check_gate(GATE, results), "critical-error rate").passed


@pytest.mark.parametrize(("wins", "passes"), [(0, False), (5, False), (6, True), (20, True)])
def test_fine_tuned_must_beat_the_baseline_with_a_small_p_value(wins, passes):
    results = passing()
    checks = check_gate(GATE, results, better_than_baseline(results, wins))
    test = check(checks, "fine-tuned vs few-shot, p-value")
    assert test.passed is passes
    assert f"only the system right on {wins}" in test.detail
    if wins == 5:
        assert test.value == pytest.approx(2 * 0.5**5)
    if wins == 6:
        assert test.value == pytest.approx(2 * 0.5**6)


def test_a_system_that_is_significantly_worse_than_the_baseline_does_not_pass():
    results = passing()
    baseline = copy.deepcopy(results["test_ood"])
    for row in results["test_ood"]["documents"][:8]:
        row["correct"] = False
    test = check(check_gate(GATE, results, baseline), "fine-tuned vs few-shot, p-value")
    assert not test.passed
    assert "only the baseline on 8" in test.detail


def test_without_a_baseline_the_test_is_not_run_and_the_gate_fails():
    checks = check_gate(GATE, passing())
    test = check(checks, "fine-tuned vs few-shot, p-value")
    assert test.value is None and not test.passed
    assert test.detail == "not run: no baseline given"
    assert failures(checks) == ["fine-tuned vs few-shot, p-value"]


def test_a_baseline_scored_on_other_documents_is_refused():
    results = passing()
    baseline = copy.deepcopy(results["test_ood"])
    baseline["documents"][0]["id"] = "other"
    with pytest.raises(ValueError, match="same documents"):
        check_gate(GATE, results, baseline)


@pytest.mark.parametrize(
    ("iid", "ood", "passes"),
    [(0.97, 0.95, True), (0.75, 0.70, True), (0.76, 0.70, False), (0.90, 0.97, True)],
)
def test_the_generalisation_gap_limit_is_inclusive_and_a_negative_gap_is_fine(iid, ood, passes):
    results = passing()
    results["test_iid"]["metrics"]["document_accuracy"]["value"] = iid
    results["test_ood"]["metrics"]["document_accuracy"]["value"] = ood
    gap = check(check_gate(GATE, results, better_than_baseline(results)), "generalisation gap")
    assert gap.value == pytest.approx(iid - ood)
    assert gap.passed is passes


@pytest.mark.parametrize(
    ("value", "passes"), [(0.90, False), (0.94, False), (0.95, True), (1.0, True)]
)
def test_review_recall_on_edge_must_be_at_least_the_minimum(value, passes):
    results = passing()
    results["edge"]["metrics"]["review_recall"]["value"] = value
    assert check(check_gate(GATE, results), "review-flag recall").passed is passes


@pytest.mark.parametrize(
    ("name", "reason"),
    [("invalid JSON rate", "invalid_output"), ("invented account rate", "unknown_account")],
)
def test_a_single_invalid_answer_in_any_split_fails_the_gate(name, reason):
    for split in GATE_SPLITS:
        results = passing()
        results[split]["documents"][3]["reasons"] = [reason]
        checks = check_gate(GATE, results, better_than_baseline(results))
        assert not check(checks, name).passed, split
        assert check(checks, name).value == pytest.approx(1 / 80)
        assert failures(checks) == [name]


@pytest.mark.parametrize("name", ["prompt-injection success", "canary leak"])
def test_the_worst_split_decides_for_injection_and_canary(name):
    metric = "injection_success_rate" if name.startswith("prompt") else "canary_leak_rate"
    for split in GATE_SPLITS:
        results = passing()
        results[split]["metrics"][metric]["value"] = 0.2
        assert not check(check_gate(GATE, results), name).passed, split
        assert check(check_gate(GATE, results), name).value == 0.2


def test_splits_without_injection_documents_are_ignored_but_not_all_of_them():
    results = passing()
    assert check(check_gate(GATE, results), "prompt-injection success").passed
    for split in GATE_SPLITS:
        results[split]["metrics"]["injection_success_rate"]["value"] = None
    injection = check(check_gate(GATE, results), "prompt-injection success")
    assert injection.value is None and not injection.passed


def test_a_split_that_was_not_run_fails_every_check_that_needs_it():
    results = passing()
    del results["edge"]
    checks = check_gate(GATE, results, better_than_baseline(passing()))
    assert failures(checks) == [
        "review-flag recall",
        "invalid JSON rate",
        "invented account rate",
        "prompt-injection success",
        "canary leak",
    ]
    assert check(checks, "review-flag recall").source == "not run: edge"
    assert check(checks, "automation rate").passed


def test_with_no_results_at_all_everything_fails():
    checks = check_gate(GATE, {})
    assert failures(checks) == NAMES
    assert all(c.value is None for c in checks)
    assert check(checks, "automation rate").source == "not run: gold"
    assert check(checks, "accuracy on automated documents").source == "not run: gold, test_ood"


def test_a_system_that_automates_nothing_has_no_accuracy_and_fails():
    results = passing()
    for split in ("gold", "test_ood"):
        results[split]["documents"] = rows(20, reviewed=20, prefix=split)
    checks = check_gate(GATE, results)
    assert check(checks, "accuracy on automated documents").value is None
    assert not check(checks, "accuracy on automated documents").passed
    assert not check(checks, "critical-error rate").passed


def test_check_semantics():
    assert Check("a", 0.6, ">=", 0.6, "").passed
    assert not Check("a", 0.59, ">=", 0.6, "").passed
    assert Check("a", 0.005, "<=", 0.005, "").passed
    assert not Check("a", 0.0051, "<=", 0.005, "").passed
    assert Check("a", 0.0, "<=", 0, "").passed
    assert Check("a", 0.049, "<", 0.05, "").passed
    assert not Check("a", 0.05, "<", 0.05, "").passed
    assert not Check("a", None, "<=", 1, "").passed


def test_the_table_shows_each_kpi_against_its_threshold():
    results = passing()
    results["gold"]["metrics"]["automation_rate"]["value"] = 0.5
    table = format_table(check_gate(GATE, results, better_than_baseline(results)))
    lines = table.splitlines()
    assert len(lines) == 11
    assert lines[0].split()[:2] == ["KPI", "value"]
    by_name = {name: line for name in NAMES for line in lines if line.startswith(name)}
    assert ">= 0.6" in by_name["automation rate"] and "FAIL" in by_name["automation rate"]
    assert "0.500" in by_name["automation rate"]
    assert ">= 0.95" in by_name["accuracy on automated documents"]
    assert "<= 0.005" in by_name["critical-error rate"]
    assert "< 0.05" in by_name["fine-tuned vs few-shot, p-value"]
    assert "<= 0.05" in by_name["generalisation gap"]
    assert "<= 0" in by_name["canary leak"]
    assert "pass" in by_name["canary leak"]
    assert "no data" in format_table(check_gate(GATE, {}))


def test_the_report_only_numbers_come_from_test_ood():
    shown = dict(report_only(passing()))
    assert shown["item F1"] == "0.900"
    assert shown["weighted error cost per 1,000 documents"] == "12.000"
    assert shown["documents per second"] == "12.000"
    assert shown["latency p50 / p95 seconds"] == "0.100 / 0.400"
    assert "not run" in dict(report_only({}))["item F1, weighted error cost, speed"]


def rule_rows(*specs):
    """Rows as (expected_review, needs_review, by_rule, reasons)."""
    return [
        {
            "id": f"d{index}",
            "expected_review": expected,
            "needs_review": flagged,
            "by_rule": by_rule,
            "reasons": reasons,
        }
        for index, (expected, flagged, by_rule, reasons) in enumerate(specs)
    ]


def test_the_rule_report_separates_rule_flags_from_model_flags():
    results = {
        "gold": {"documents": rule_rows((False, False, False, []))},
        "edge": {
            "documents": rule_rows(
                (True, True, True, ["foreign_currency"]),
                (True, True, True, ["no_total"]),
                (True, True, False, ["model_flagged"]),
                (True, False, False, []),
            )
        },
    }
    shown = dict(rule_report(results))
    assert (
        shown["documents flagged by a pipeline rule"]
        == "2 of 5 (40.0%), 1 of them for foreign currency"
    )
    assert shown["review recall on edge, with / without the rules"] == "0.750 / 0.250"


def test_results_from_before_the_rule_report_say_that_it_is_not_recorded():
    old = {
        "edge": {
            "documents": [
                {"id": "d0", "expected_review": True, "needs_review": True, "reasons": []}
            ]
        }
    }
    assert dict(rule_report(old)) == {"flagged by a pipeline rule": "not recorded in these results"}
    assert "not recorded" in dict(rule_report({}))["flagged by a pipeline rule"]


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """Real runs against the fake server: a perfect system on all four splits, and a
    baseline that books every third document to a wrong account, on test_ood."""
    out = tmp_path_factory.mktemp("results")

    def run(server, system, split, answer=None):
        server.handler = Model(
            load_documents("tax-booking", split), load_canaries("tax-booking"), answer=answer
        )
        settings = RunSettings(
            "tax-booking", system, split, server.url, "tax-booking", threshold=0.0, workers=4
        )
        write_results(evaluate(settings), results_file(out, "tax-booking", system, split))

    with FakeServer() as server:
        for split in GATE_SPLITS:
            run(server, "good", split)
        run(server, "weak", "test_ood", wrong_account_with_low_confidence())
        run(server, "weak", "gold", wrong_account_with_low_confidence())
    return out


def test_a_perfect_system_passes_the_gate_from_real_results_files(results_dir, capsys):
    code = main(["--system", "good", "--baseline", "weak", "--out-dir", str(results_dir)])
    out = capsys.readouterr().out
    assert code == 0
    assert "gate for good against weak" in out
    assert "gate: PASS (10 of 10 checks)" in out
    assert "report only: item F1: 1.000" in out
    assert out.count("pass") >= 10


def test_the_same_system_without_a_baseline_fails_only_the_comparison(results_dir, capsys):
    code = main(["--system", "good", "--out-dir", str(results_dir)])
    out = capsys.readouterr().out
    assert code == 1
    assert "gate: FAIL (1 of 10 checks)" in out
    assert "not run: no baseline given" in out


def test_a_baseline_without_results_is_named_in_the_table(results_dir, capsys):
    code = main(["--system", "good", "--baseline", "ghost", "--out-dir", str(results_dir)])
    out = capsys.readouterr().out
    assert code == 1
    assert "not run: no test_ood results for ghost" in out


def test_a_weak_system_fails_the_gate_with_the_numbers_shown(results_dir, capsys):
    code = main(["--system", "weak", "--baseline", "good", "--out-dir", str(results_dir)])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in out
    assert "not run: edge" in out or "not run: test_iid" in out
    assert "gate: FAIL" in out


def test_a_system_with_no_results_fails_with_a_message_per_check(results_dir, capsys):
    code = main(["--system", "never-run", "--out-dir", str(results_dir)])
    out = capsys.readouterr().out
    assert code == 1
    assert "gate: FAIL (10 of 10 checks)" in out
    assert "not run: gold" in out


def test_the_system_name_is_required():
    with pytest.raises(SystemExit):
        main([])

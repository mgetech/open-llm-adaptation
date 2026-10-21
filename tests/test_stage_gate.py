"""The stage report: the gate checks that dev and dev_real can show."""

import json

import pytest

from adapt.config import load_platform
from adapt.eval.runner import results_file
from adapt.eval.stage_gate import checks, load_margin, main, margin_report, report, verdict

GATE = load_platform("tax-booking").config.gate


def split(**values):
    """A results file with passing numbers, changed by `values`."""
    metrics = {
        "automation_rate": 0.8,
        "accuracy_on_automated": 0.97,
        "critical_error_rate": 0.0,
        "review_recall": 1.0,
        "invalid_json_rate": 0.0,
        "invented_account_rate": 0.0,
        "injection_success_rate": 0.0,
        "injection_unexpected_rate": 0.1,
        "canary_leak_rate": 0.0,
        "document_accuracy": 0.8,
        **values,
    }
    rows = [
        {"needs_review": False, "severity": None},
        {"needs_review": False, "severity": "critical"},
        {"needs_review": True, "severity": None},
    ]
    return {
        "metrics": {name: {"value": value} for name, value in metrics.items()},
        "manifest": {"threshold": 0.98},
        "confidence_threshold": {"value": 0.98, "source": "tuned on dev"},
        "documents": rows,
    }


def test_each_check_has_a_name_a_metric_an_operator_and_the_limit_of_the_gate():
    shown = {name: (metric, op, limit) for name, metric, op, limit in checks(GATE)}
    assert shown["automation rate"] == ("automation_rate", ">=", GATE.automation_rate_min)
    assert shown["critical-error rate"] == ("critical_error_rate", "<=", 0.005)
    assert len(shown) == 8


@pytest.mark.parametrize(
    ("value", "operator", "limit", "expected"),
    [
        (0.6, ">=", 0.6, "pass"),
        (0.59, ">=", 0.6, "FAIL"),
        (0.005, "<=", 0.005, "pass"),
        (0.0051, "<=", 0.005, "FAIL"),
        (0.0, "<=", 0.0, "pass"),
        (None, "<=", 0.0, "n/a"),
    ],
)
def test_a_value_passes_at_its_limit_and_a_missing_value_is_not_judged(
    value, operator, limit, expected
):
    assert verdict(value, operator, limit) == expected


def test_passing_numbers_pass_the_stage_with_one_line_per_split():
    text, passed = report(GATE, {"dev": split(), "dev_real": split()}, "demo")
    assert passed and "stage: PASS" in text
    assert "stage report for demo: dev and dev_real, not the claim" in text
    assert "confidence threshold 0.9800 (tuned on dev)" in text
    assert "dev: 1 critical in 2 automated of 3, document accuracy 0.800" in text
    assert "generalisation gap: needs test_iid and test_ood" in text


def test_a_check_that_fails_on_either_split_fails_the_stage_and_is_named():
    text, passed = report(
        GATE, {"dev": split(), "dev_real": split(critical_error_rate=0.02)}, "demo"
    )
    assert not passed
    assert "critical-error rate" in text.splitlines()[-1] and "stage: FAIL" in text
    line = next(line for line in text.splitlines() if line.startswith("critical-error rate"))
    assert "0.0000 pass" in line and "0.0200 FAIL" in line


def test_a_split_without_injection_documents_is_not_judged_on_injection():
    text, passed = report(GATE, {"dev_real": split(injection_success_rate=None)}, "demo")
    assert passed
    assert "no documents n/a" in injection_line(text)


def test_the_old_injection_figure_is_not_read_as_the_new_one():
    old = split(injection_success_rate=0.5)
    del old["metrics"]["injection_unexpected_rate"]
    text, passed = report(GATE, {"dev": old}, "demo")
    assert passed  # not judged, so it neither passes nor fails
    assert "no documents n/a" in injection_line(text)


def injection_line(text):
    return next(line for line in text.splitlines() if line.startswith("prompt-injection"))


def write(out_dir, system, name, data):
    path = results_file(out_dir, "tax-booking", system, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_the_command_reads_dev_and_dev_real_and_returns_one_when_a_check_fails(tmp_path, capsys):
    write(tmp_path, "demo", "dev", split())
    write(tmp_path, "demo", "dev_real", split(accuracy_on_automated=0.9))
    arguments = ["--system", "demo", "--round", "3", "--out-dir", str(tmp_path)]
    assert main(arguments) == 1
    out = capsys.readouterr().out
    assert "0.9000 FAIL" in out and "stage: FAIL (accuracy on automated documents)" in out


def test_the_command_works_with_only_dev_and_returns_zero_when_everything_passes(tmp_path, capsys):
    write(tmp_path, "demo", "dev", split())
    assert main(["--system", "demo", "--round", "3", "--out-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("stage report for demo: dev, not")


def test_a_system_without_results_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no dev or dev_real results for demo"):
        main(["--system", "demo", "--round", "3", "--out-dir", str(tmp_path)])


MARGIN = {
    "dev": {"critical_error_rate_max": 0.0025},
    "dev_real": {"automation_rate_min": 0.7, "accuracy_on_automated_min": 0.97},
}


def test_the_committed_margin_is_the_one_fixed_before_the_runs():
    assert load_margin() == {
        "dev": {"critical_error_rate_max": 0.0025},
        "dev_real": {"automation_rate_min": 0.70, "accuracy_on_automated_min": 0.97},
    }


def test_a_configuration_that_meets_every_margin_number_meets_the_margin():
    results = {
        "dev": split(critical_error_rate=0.0),
        "dev_real": split(automation_rate=0.72, accuracy_on_automated=0.975),
    }
    text, met = margin_report(MARGIN, results)
    assert met and text.endswith("margin: MET")
    assert "dev critical-error rate <= 0.0025: 0.0000 pass" in text


def test_one_number_short_of_the_margin_means_it_is_not_met_even_if_the_gate_passes():
    results = {
        "dev": split(critical_error_rate=0.004),
        "dev_real": split(automation_rate=0.72, accuracy_on_automated=0.975),
    }
    assert report(GATE, results, "demo")[1]  # the gate checks all pass
    text, met = margin_report(MARGIN, results)
    assert not met and "dev critical-error rate <= 0.0025: 0.0040 FAIL" in text
    assert text.endswith("margin: NOT MET")


def test_a_margin_number_is_met_exactly_at_its_limit_and_a_missing_split_is_not_met():
    at_limit = {
        "dev": split(critical_error_rate=0.0025),
        "dev_real": split(automation_rate=0.7, accuracy_on_automated=0.97),
    }
    assert margin_report(MARGIN, at_limit)[1]
    assert not margin_report(MARGIN, {"dev": split(critical_error_rate=0.0)})[1]


def test_the_command_checks_the_margin_and_returns_one_when_it_is_not_met(tmp_path, capsys):
    write(tmp_path, "demo", "dev", split(critical_error_rate=0.004))
    write(tmp_path, "demo", "dev_real", split(automation_rate=0.72, accuracy_on_automated=0.975))
    arguments = ["--system", "demo", "--round", "3", "--out-dir", str(tmp_path)]
    assert main(arguments) == 0
    assert main([*arguments, "--margin"]) == 1
    assert "margin: NOT MET" in capsys.readouterr().out

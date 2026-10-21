"""The coverage and critical-error report on dev and dev_real."""

import pytest

from adapt.config import load_platform
from adapt.eval.threshold import tune_threshold
from adapt.eval.threshold_report import describe, format_split, main

PLATFORM = load_platform("tax-booking")


def row(confidence, right, severity=None):
    return {
        "expected_review": False,
        "reasons": [],
        "confidence": confidence,
        "answer_correct": right,
        "severity": severity,
    }


ROWS = [row(0.99, True), row(0.95, True), row(0.9, False, "critical"), row(0.8, True)]


def test_each_threshold_shows_the_automated_documents_and_their_critical_errors():
    text = format_split("dev", 4, ROWS, 0.95, PLATFORM, "chosen on dev")
    assert text.startswith("dev (4 documents)")
    line = next(line for line in text.splitlines() if line.strip().startswith("0.9000"))
    assert "3 of 3" not in line and "1 of 3" in line  # 0.99, 0.95 and 0.9 are automated
    chosen = next(line for line in text.splitlines() if "<- chosen on dev" in line)
    assert chosen.strip().startswith("0.9500") and "0 of 2" in chosen


def test_the_chosen_threshold_is_checked_against_the_gate_limits():
    text = format_split("dev_real", 4, ROWS, 0.95, PLATFORM, "from dev, not chosen again")
    assert "at 0.9500: automation 0.500 FAIL, accuracy 1.000 pass, critical 0.000 pass" in text
    worse = format_split("dev_real", 4, ROWS, 0.0, PLATFORM, "x")
    assert "critical 0.250 FAIL" in worse


def test_the_description_says_which_targets_the_threshold_meets():
    assert "meets the accuracy target and the limit" in describe(tune_threshold(ROWS, 0.75, 0.0))
    only_accuracy = [row(0.99, False, "critical"), row(0.9, True), row(0.8, True)]
    assert "limit on critical errors is not" in describe(tune_threshold(only_accuracy, 0.6, 0.0))
    assert "no confidence limit" in describe(
        tune_threshold([row(0.9, False, "critical")], 0.95, 0.0)
    )


def test_the_report_on_the_saved_round_3_results_shows_both_splits(capsys):
    assert main(["--system", "qwen3-4b-sft-r3-0", "--round", "3"]) == 0
    out = capsys.readouterr().out
    assert "threshold chosen on dev:" in out
    assert "dev (300 documents)" in out and "dev_real (96 documents)" in out
    assert "<- chosen on dev" in out and "<- from dev, not chosen again" in out


def test_a_system_without_saved_results_is_refused(capsys):
    with pytest.raises(SystemExit, match="no saved dev results"):
        main(["--system", "nothing", "--round", "3"])


def test_the_report_can_choose_the_threshold_on_both_splits_together(capsys):
    arguments = ["--system", "qwen3-4b-sft-r3-0", "--round", "3", "--tune-on", "dev", "dev_real"]
    assert main(arguments) == 0
    out = capsys.readouterr().out
    assert "threshold chosen on dev and dev_real:" in out
    assert out.count("<- chosen on dev and dev_real") == 2
    assert "not chosen again" not in out


def test_the_description_names_the_target_that_is_missed():
    only_critical = [row(0.99, False), row(0.9, False), row(0.8, True)]
    assert "limit on critical errors is met, the accuracy target is not" in describe(
        tune_threshold(only_critical, 0.95, 0.0)
    )

"""The post-hoc re-scoring of the shipped configuration on gold and gold_v2: labelled, not a claim,
the threshold from dev and dev_real, and gold_v3 never read."""

import json
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.eval.post_hoc_gold import GOLD_SETS, LABEL, OUT, TUNE_ON, build
from adapt.eval.rescore import tune_pooled
from adapt.eval.runner import load_documents, results_file

ROUND_THREE = Path(__file__).resolve().parents[1] / "results" / "round3" / "eval"
SYSTEM = "qwen3-4b-sft-r3-0"


@pytest.fixture(scope="module")
def report():
    return build("tax-booking", SYSTEM, 3, ROUND_THREE)


def test_the_report_says_it_is_post_hoc_and_not_held_out(report):
    assert report["post_hoc"] is True and report["claim"] is False
    assert report["label"] == LABEL
    assert "POST-HOC, NOT A CLAIM" in LABEL and "not held out" in LABEL


def test_gold_v3_is_not_read(report):
    assert GOLD_SETS == ("gold", "gold_v2")
    assert list(report["splits"]) == list(GOLD_SETS)
    assert not any("gold_v3" in name for name in report["inputs_sha256"])


def test_the_threshold_is_the_one_chosen_on_dev_and_dev_real(report):
    platform = load_platform("tax-booking")
    sources = []
    for split in TUNE_ON:
        saved = json.loads(
            results_file(ROUND_THREE, "tax-booking", SYSTEM, split).read_text("utf-8")
        )
        sources.append((saved, load_documents("tax-booking", split, data_round=3)))
    assert report["threshold"]["value"] == tune_pooled(sources, platform).threshold
    assert report["threshold"]["chosen_on"] == "dev and dev_real"


def test_the_counts_agree_with_the_rates(report):
    for data in report["splits"].values():
        metrics = data["metrics"]
        assert metrics["automation_rate"] == pytest.approx(data["automated"] / data["automatable"])
        assert metrics["accuracy_on_automated"] == pytest.approx(
            data["automated_correct"] / data["automated"]
        )
        assert data["critical_errors"] == data["severity_counts"]["critical"]


def test_the_committed_report_is_what_the_saved_results_give(report):
    committed = json.loads(OUT.read_text(encoding="utf-8"))
    for key in ("label", "round", "system", "threshold", "inputs_sha256", "splits"):
        assert committed[key] == report[key]

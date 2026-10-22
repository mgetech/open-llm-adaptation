"""The post-hoc re-scoring of the round 3 claim run: labelled, not a claim, and the threshold comes
from dev only."""

import json
import shutil
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.eval.post_hoc import GOLD_SETS, LABEL, SPLITS, build, main
from adapt.eval.rescore import tune_on_dev
from adapt.eval.runner import RESULTS_ROOT, load_documents, results_file

ROUND_THREE = Path(__file__).resolve().parents[1] / "results" / "round3" / "eval"
SYSTEM = "qwen3-4b-sft-r3-0"
BASELINE = "qwen3-4b-fewshot-r3"
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


@pytest.fixture(scope="module")
def report():
    return build("tax-booking", SYSTEM, BASELINE, 3, ROUND_THREE)


def test_the_report_says_it_is_post_hoc_and_not_a_claim(report):
    data, _ = report
    assert data["post_hoc"] is True and data["claim"] is False
    assert data["label"] == LABEL
    assert "POST-HOC, NOT A CLAIM" in LABEL and "already evaluated" in LABEL


def test_every_gate_check_is_reported_for_each_gold_set(report):
    data, texts = report
    assert list(data["gates"]) == list(GOLD_SETS) and GOLD_SETS[0] == "gold_v2"
    for gold, gate in data["gates"].items():
        assert [c["name"] for c in gate["checks"]] == NAMES
        assert gate["total"] == 10 and gate["passed"] == sum(c["passed"] for c in gate["checks"])
        assert gate["result"] == ("PASS" if gate["passed"] == 10 else "FAIL")
        assert gold in texts and "critical-error rate" in texts[gold]


def test_the_threshold_is_the_one_chosen_on_dev(report):
    data, _ = report
    platform = load_platform("tax-booking")
    dev = json.loads(results_file(ROUND_THREE, "tax-booking", SYSTEM, "dev").read_text("utf-8"))
    tuned = tune_on_dev(dev, load_documents("tax-booking", "dev", data_round=3), platform)
    assert data["threshold"]["value"] == tuned.threshold
    assert data["threshold"]["chosen_on"] == "dev"
    assert {d["threshold"] for d in data["splits"].values()} == {tuned.threshold}


def test_the_test_splits_cannot_change_the_threshold(tmp_path, report):
    """The threshold stays the same when the answers of the other splits are different."""
    for split in SPLITS:
        copied = results_file(tmp_path, "tax-booking", SYSTEM, split)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(results_file(ROUND_THREE, "tax-booking", SYSTEM, split), copied)
    for name in (SYSTEM, BASELINE):
        for split in ("dev", "test_ood") if name == BASELINE else ():
            target = results_file(tmp_path, "tax-booking", name, split)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(results_file(ROUND_THREE, "tax-booking", name, split), target)
    for split in ("test_iid", "test_ood", "edge", "gold_v2"):
        path = results_file(tmp_path, "tax-booking", SYSTEM, split)
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["documents"]:
            row["confidence"] = 0.01
        path.write_text(json.dumps(data), encoding="utf-8")
    changed, _ = build("tax-booking", SYSTEM, BASELINE, 3, tmp_path)
    assert changed["threshold"]["value"] == report[0]["threshold"]["value"]
    assert changed["splits"]["test_ood"]["metrics"]["automation_rate"] == 0.0


def test_the_command_writes_the_report_and_prints_the_label(tmp_path, capsys):
    out = tmp_path / "post_hoc" / "round3_rescored.json"
    arguments = ["--system", SYSTEM, "--baseline", BASELINE, "--round", "3", "--out", str(out)]
    lock = (RESULTS_ROOT / "final.lock.json").read_text(encoding="utf-8")
    assert main(arguments) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("POST-HOC, NOT A CLAIM") and "post-hoc gate with gold_v2" in printed
    assert printed.rstrip().endswith(f"report: {out}")
    assert json.loads(out.read_text(encoding="utf-8"))["post_hoc"] is True
    # it reads saved results and makes no claim
    assert (RESULTS_ROOT / "final.lock.json").read_text(encoding="utf-8") == lock


def test_a_missing_results_file_is_refused_with_a_message(tmp_path):
    with pytest.raises(SystemExit, match="error"):
        main(
            [
                "--system",
                "nothing",
                "--baseline",
                BASELINE,
                "--round",
                "3",
                "--in-dir",
                str(tmp_path),
            ]
        )


def test_the_committed_report_is_labelled_post_hoc():
    path = RESULTS_ROOT / "post_hoc" / "round3_rescored.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["post_hoc"] is True and data["claim"] is False and data["label"] == LABEL
    assert data["system"] == SYSTEM and data["round"] == 3
    assert set(data["gates"]) == set(GOLD_SETS)

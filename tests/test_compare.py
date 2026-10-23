"""The comparison of two results files."""

import copy
import json
import shutil
import subprocess

import pytest

from adapt.eval import compare as compare_module
from adapt.eval.compare import compare, main, render
from adapt.eval.runner import ROOT

ADAPTER = "a" * 64
MANIFEST = {
    "model_revision": "r1",
    "adapter_hash": ADAPTER,
    "prompt_sha256": "p" * 64,
    "threshold": 0.98,
    "few_shot": False,
    "gpu": "Tesla T4",
    "vllm_version": "0.30.0",
    "git_commit": "abc",
}


def interval(value, low, high):
    return {"value": value, "low": low, "high": high}


def results():
    return {
        "split": "dev",
        "manifest": dict(MANIFEST),
        "metrics": {
            "document_accuracy": interval(0.8, 0.7, 0.9),
            "automation_rate": interval(0.6, 0.5, 0.7),
            "accuracy_on_automated": interval(0.95, 0.9, 1.0),
            "critical_error_rate": interval(0.0, 0.0, 0.0),
            "review_recall": interval(1.0, 1.0, 1.0),
            "injection_success_rate": interval(None, None, None),
        },
        "documents": [
            {"id": "d1", "output": {"x": 1}, "needs_review": False, "confidence": 0.999},
            {"id": "d2", "output": {"x": 2}, "needs_review": True, "confidence": 0.5},
            {"id": "d3", "output": {"x": 3}, "needs_review": True, "confidence": None},
        ],
    }


@pytest.fixture
def pair():
    first = results()
    return first, copy.deepcopy(first)


def test_the_same_run_on_another_machine_passes(pair):
    first, second = pair
    second["manifest"].update(gpu="Tesla T4 (VM)", vllm_version="0.30.1", git_commit="def")
    report = compare(first, second)
    assert report["passed"] is True
    assert report["identical_outputs"] == 3 and report["same_review_decision"] == 3
    assert report["largest_confidence_difference"] == 0
    assert report["manifest"]["gpu"]["same"] is False


def test_a_changed_output_is_counted_and_the_confidence_difference_is_the_largest(pair):
    first, second = pair
    second["documents"][0]["output"] = {"x": 9}
    second["documents"][1]["confidence"] = 0.4
    second["documents"][1]["needs_review"] = False
    report = compare(first, second)
    assert report["identical_outputs"] == 2 and report["same_review_decision"] == 2
    assert report["largest_confidence_difference"] == pytest.approx(0.1)


@pytest.mark.parametrize("key", ["model_revision", "adapter_hash", "prompt_sha256", "threshold"])
def test_another_adapter_prompt_revision_or_threshold_fails(pair, key):
    first, second = pair
    second["manifest"][key] = "other" if key != "threshold" else 0.5
    assert compare(first, second)["passed"] is False


def test_a_metric_outside_the_interval_of_the_reference_fails(pair):
    first, second = pair
    second["metrics"]["automation_rate"]["value"] = 0.8
    report = compare(first, second)
    assert report["metrics"]["automation_rate"]["inside"] is False
    assert report["passed"] is False


def test_other_documents_fail(pair):
    first, second = pair
    second["documents"][2]["id"] = "d4"
    report = compare(first, second)
    assert report["same_documents"] is False and report["passed"] is False


def test_another_split_fails(pair):
    first, second = pair
    second["split"] = "dev_real"
    assert compare(first, second)["passed"] is False


def test_a_metric_without_a_value_is_left_out(pair):
    first, second = pair
    assert "injection_success_rate" not in compare(first, second)["metrics"]


def test_results_without_any_metric_do_not_pass(pair):
    first, second = pair
    first["metrics"], second["metrics"] = {}, {}
    assert compare(first, second)["passed"] is False


def test_the_text_says_pass_or_fail_and_marks_what_differs(pair):
    first, second = pair
    second["manifest"]["prompt_sha256"] = "q" * 64
    text = render(compare(first, second))
    assert text.splitlines()[-1] == "FAIL" and "prompt_sha256: DIFFERENT" in text
    assert render(compare(first, copy.deepcopy(first))).splitlines()[-1] == "PASS"


def test_the_entry_point_prints_writes_and_exits_by_the_result(pair, tmp_path, capsys):
    first, second = pair
    (tmp_path / "a.json").write_text(json.dumps(first), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(second), encoding="utf-8")
    out = tmp_path / "out" / "compare.json"
    arguments = ["--first", str(tmp_path / "a.json"), "--second", str(tmp_path / "b.json")]
    assert main([*arguments, "--out", str(out)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True
    second["manifest"]["threshold"] = 0.1
    (tmp_path / "b.json").write_text(json.dumps(second), encoding="utf-8")
    assert main(arguments) == 1


def test_a_missing_file_is_an_error_with_a_message(tmp_path):
    with pytest.raises(SystemExit, match="error"):
        main(["--first", str(tmp_path / "none.json"), "--second", str(tmp_path / "none.json")])


def test_a_results_file_of_the_repository_equals_itself():
    path = ROOT / "results/round3/rescored/tax-booking/qwen3-4b-sft-r3-0/dev_real.json"
    report = compare(compare_module.load(path), compare_module.load(path))
    assert report["passed"] is True and report["identical_outputs"] == report["documents"] == 96


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_the_make_target_passes_its_files_on():
    def planned(*variables):
        return subprocess.run(
            ["make", "--dry-run", "--no-print-directory", "compare", *variables],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    command = planned("FIRST=a.json", "SECOND=b.json")
    assert "-m adapt.eval.compare --first a.json --second b.json" in command
    assert "--out" not in command and "--out c.json" in planned("FIRST=a", "SECOND=b", "OUT=c.json")

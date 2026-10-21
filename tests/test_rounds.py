"""Choosing the round of the data for training, evaluation and the baseline.

These tests use the real data folder: the last round is the current one and round 1 is kept in
`data/tax-booking/round1/`. They use the fake server, so they need no GPU.
"""

import copy
import json
import shutil
import subprocess

import pytest

from adapt.baseline import main as baseline_main
from adapt.baseline import system_name as baseline_name
from adapt.config import load_platform
from adapt.data.rounds import (
    current_round,
    eval_dir,
    generator_version,
    resolve_round,
    round_folder,
    round_tag,
    runs_dir,
)
from adapt.eval.gate import check_gate
from adapt.eval.runner import DATA_DIR, RESULTS_DIR, ROOT, load_documents, results_file, split_path
from adapt.train.evaluate import main as evaluate_main
from adapt.train.evaluate import system_name

GATE = load_platform("tax-booking").config.gate
needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def test_the_current_round_is_one_more_than_the_kept_rounds():
    manifest = json.loads((ROOT / "results" / "data_manifest.json").read_text("utf-8"))
    assert current_round() == len(manifest["rounds"]) + 1 >= 2


def test_a_run_without_a_round_uses_the_current_one():
    assert resolve_round(None) == current_round()
    assert resolve_round(1) == 1


@pytest.mark.parametrize("wrong", [0, -1, 99])
def test_a_round_that_does_not_exist_is_refused(wrong):
    with pytest.raises(ValueError, match="does not exist"):
        resolve_round(wrong)


def test_round_one_has_the_plain_names_and_later_rounds_add_their_number():
    assert round_tag(1) == "" and round_tag(2) == "-r2" and round_tag(10) == "-r10"
    assert system_name("qwen3-4b", 0, data_round=1) == "qwen3-4b-sft-0"
    assert system_name("qwen3-4b", 0, data_round=2) == "qwen3-4b-sft-r2-0"
    assert system_name("qwen3-4b", 0, max_steps=3, data_round=2) == "qwen3-4b-sft-r2-0-steps3"
    assert baseline_name("qwen3-4b") == "qwen3-4b-fewshot"
    assert baseline_name("qwen3-4b", 2) == "qwen3-4b-fewshot-r2"


def test_every_round_is_in_its_own_folder_and_the_current_one_is_the_last():
    current = current_round()
    assert round_folder("tax-booking", None) == DATA_DIR / "tax-booking" / f"round{current}"
    assert round_folder("tax-booking", current) == DATA_DIR / "tax-booking" / f"round{current}"
    assert round_folder("tax-booking", 1) == DATA_DIR / "tax-booking" / "round1"
    assert split_path("tax-booking", "test_ood", data_round=1).parent.name == "round1"
    assert split_path("tax-booking", "test_ood").parent.name == f"round{current}"


def test_the_gold_set_does_not_depend_on_the_round():
    assert split_path("tax-booking", "gold", data_round=1) == split_path("tax-booking", "gold")


def test_the_rounds_hold_other_documents_under_the_same_ids():
    first = load_documents("tax-booking", "test_ood", data_round=1)
    later = load_documents("tax-booking", "test_ood")
    assert [d.id for d in first] == [d.id for d in later]
    assert [d.text for d in first] != [d.text for d in later]


def test_the_generator_version_of_each_round():
    assert generator_version(1) == 1
    assert generator_version(None) == generator_version(current_round()) >= 2


def test_a_round_that_is_not_the_first_is_made_by_a_newer_generator():
    manifest = json.loads((ROOT / "results" / "data_manifest.json").read_text("utf-8"))
    assert manifest["generator"] > manifest["rounds"][0].get("generator", 1)


def make_adapter(adapters_dir, name):
    folder = adapters_dir / "tax-booking" / name
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    return folder


def test_an_adapter_is_evaluated_on_the_data_of_its_round(tmp_path):
    make_adapter(tmp_path, f"qwen3-1.7b{round_tag(current_round())}-0")
    make_adapter(tmp_path, "qwen3-1.7b-0")
    base = ["--model", "qwen3-1.7b", "--dry-run", "--splits", "dev", "test_ood"]
    base += ["--adapters-dir", str(tmp_path)]
    assert evaluate_main([*base, "--out-dir", str(tmp_path / "later")]) == 0
    assert evaluate_main([*base, "--round", "1", "--out-dir", str(tmp_path / "first")]) == 0

    later = json.loads(
        results_file(
            tmp_path / "later",
            "tax-booking",
            f"qwen3-1.7b-sft{round_tag(current_round())}-0",
            "test_ood",
        ).read_text(encoding="utf-8")
    )
    first = json.loads(
        results_file(tmp_path / "first", "tax-booking", "qwen3-1.7b-sft-0", "test_ood").read_text(
            encoding="utf-8"
        )
    )
    assert later["manifest"]["data_round"] == current_round()
    assert later["manifest"]["data_generator"] == generator_version(None)
    assert first["manifest"]["data_round"] == 1 and first["manifest"]["data_generator"] == 1
    manifest = json.loads((ROOT / "results" / "data_manifest.json").read_text("utf-8"))
    assert later["manifest"]["data_sha256"] == manifest["splits"]["test_ood"]["sha256"]
    assert first["manifest"]["data_sha256"] == manifest["rounds"][0]["splits"]["test_ood"]["sha256"]


def test_an_adapter_of_another_round_is_not_found(tmp_path, capsys):
    make_adapter(tmp_path, "qwen3-1.7b-0")  # round 1 only
    arguments = ["--model", "qwen3-1.7b", "--dry-run", "--adapters-dir", str(tmp_path)]
    with pytest.raises(
        SystemExit, match=f"no adapter in .*qwen3-1.7b{round_tag(current_round())}-0"
    ):
        evaluate_main(arguments)


def test_the_baseline_of_a_later_round_has_its_own_name_and_can_be_limited_to_a_split(tmp_path):
    arguments = ["--model", "qwen3-1.7b", "--dry-run", "--out-dir", str(tmp_path)]
    assert baseline_main([*arguments, "--splits", "dev", "test_ood"]) == 0
    folder = tmp_path / "tax-booking" / f"qwen3-1.7b-fewshot{round_tag(current_round())}"
    assert sorted(path.name for path in folder.iterdir()) == ["dev.json", "test_ood.json"]
    results = json.loads((folder / "test_ood.json").read_text(encoding="utf-8"))
    assert results["manifest"]["data_round"] == current_round()
    assert results["manifest"]["few_shot"] is True


def test_the_baseline_refuses_an_unknown_split(tmp_path):
    arguments = ["--model", "qwen3-1.7b", "--dry-run", "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit, match="unknown split 'train'"):
        baseline_main([*arguments, "--splits", "train"])


def passing_pair():
    """Results of a system and of a baseline that was run on the same test_ood file."""
    rows = [
        {"id": f"o{n}", "correct": True, "severity": None, "needs_review": False, "reasons": []}
        for n in range(40)
    ]
    metrics = {"document_accuracy": {"value": 0.9}}
    system = {"manifest": {"data_sha256": "aaa"}, "metrics": metrics, "documents": rows}
    baseline = copy.deepcopy(system)
    for row in baseline["documents"][:8]:
        row["correct"] = False
    return system, baseline


def test_a_baseline_of_other_data_is_not_compared():
    system, baseline = passing_pair()
    results = {"test_ood": system}
    test = next(c for c in check_gate(GATE, results, baseline) if c.name.startswith("fine-tuned"))
    assert test.passed

    baseline["manifest"]["data_sha256"] = "bbb"
    test = next(c for c in check_gate(GATE, results, baseline) if c.name.startswith("fine-tuned"))
    assert not test.passed and test.value is None
    assert test.detail == "not run: the baseline ran on other test_ood data"


@needs_make
@pytest.mark.parametrize(
    ("target", "variables"),
    [
        ("train", ["MODEL=qwen3-4b"]),
        ("eval-sft", ["MODEL=qwen3-4b"]),
        ("baseline", ["MODEL=qwen3-4b"]),
        ("eval", ["SYSTEM=x"]),
        ("gate", ["SYSTEM=x"]),
    ],
)
def test_make_passes_the_round_on(target, variables):
    def planned(*extra):
        result = subprocess.run(
            ["make", "--dry-run", target, *variables, *extra],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return " ".join(result.stdout.replace("\\\n", " ").split())

    assert "--round 2" in planned("ROUND=2")
    assert "--round" not in planned()


def test_the_results_of_each_round_have_their_own_folders():
    assert eval_dir(1).as_posix() == "results/round1/eval"
    assert runs_dir(2).as_posix() == "results/round2/runs"
    assert eval_dir().as_posix() == f"results/round{current_round()}/eval"
    assert eval_dir() == RESULTS_DIR
    with pytest.raises(ValueError, match="does not exist"):
        eval_dir(99)


def test_the_round_one_results_are_in_their_folder_and_nowhere_else():
    first = ROOT / "results" / "round1"
    assert (first / "eval" / "tax-booking" / "qwen3-4b-sft-0" / "test_ood.json").exists()
    assert (first / "runs" / "tax-booking-qwen3-4b-0.json").exists()
    assert not (ROOT / "results" / "eval").exists() and not (ROOT / "results" / "runs").exists()


def test_the_gate_finds_the_results_of_a_round_by_its_number(capsys):
    from adapt.eval.gate import main as gate

    code = gate(["--system", "qwen3-4b-sft-0", "--baseline", "qwen3-4b-fewshot", "--round", "1"])
    out = capsys.readouterr().out
    # The round 1 results have only the old injection figure, which the gate does not read.
    assert code == 1 and "gate: FAIL (3 of 10 checks)" in out
    assert "old definition in" in out and "run make rescore" in out
    # Round 2 has no results yet, so every check is missing.
    code = gate(["--system", "qwen3-4b-sft-0", "--round", "2"])
    assert code == 1 and "gate: FAIL (10 of 10 checks)" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="round 99 does not exist"):
        gate(["--system", "qwen3-4b-sft-0", "--round", "99"])


def test_a_run_writes_into_the_results_folder_of_its_round(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arguments = ["--model", "qwen3-1.7b", "--dry-run", "--splits", "dev", "--round", "1"]
    assert baseline_main(arguments) == 0
    written = tmp_path / "results" / "round1" / "eval" / "tax-booking" / "qwen3-1.7b-fewshot"
    assert [path.name for path in written.iterdir()] == ["dev.json"]

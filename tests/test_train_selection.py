"""Choosing the checkpoint of a run. A smoke run leaves real checkpoints on the CPU and the fake
server scores them, so these tests need the `gpu` dependency group but no GPU."""

import json
import shutil

import pytest

from adapt.eval.runner import hash_directory, hash_file, results_file, split_path
from adapt.models import load_registry
from adapt.train import selection as selection_module
from adapt.train.run import TrainSettings, run
from adapt.train.selection import (
    ADAPTER_FILES,
    METHOD,
    candidate_system,
    choose,
    export_adapter,
    main,
    select_checkpoint,
)

for library in ("torch", "transformers", "trl", "peft", "datasets", "tokenizers"):
    pytest.importorskip(library)

PATH = ("tax-booking", "smoke-sft-0")


def candidate(step, rate, accuracy):
    return {"step": step, "critical_error_rate": rate, "document_accuracy": accuracy}


def test_the_lowest_critical_error_rate_wins():
    found = choose([candidate(150, 0.05, 0.9), candidate(225, 0.02, 0.8), candidate(300, 0.04, 1)])
    assert found["step"] == 225


def test_a_tie_goes_to_the_higher_document_accuracy_and_then_to_the_later_step():
    assert choose([candidate(150, 0.0, 0.8), candidate(225, 0.0, 0.9)])["step"] == 225
    assert choose([candidate(150, 0.0, 0.9), candidate(225, 0.0, 0.8)])["step"] == 150
    assert choose([candidate(150, 0.0, 0.9), candidate(300, 0.0, 0.9)])["step"] == 300


def test_a_rate_that_cannot_be_computed_is_the_worst_one():
    assert choose([candidate(150, None, 0.9), candidate(225, 0.5, 0.1)])["step"] == 225
    assert choose([candidate(150, None, None), candidate(225, None, 0.1)])["step"] == 225


def test_the_candidates_are_named_after_the_system_and_the_step():
    assert candidate_system("qwen3-4b-sft-r3-0", 150) == "qwen3-4b-sft-r3-0-ckpt150"


def test_an_exported_checkpoint_has_the_adapter_files_and_nothing_else(tmp_path):
    checkpoint = tmp_path / "checkpoint-150"
    checkpoint.mkdir()
    for name in (*ADAPTER_FILES, "optimizer.pt", "scheduler.pt", "trainer_state.json"):
        (checkpoint / name).write_bytes(name.encode())
    target = export_adapter(checkpoint, tmp_path / "run-ckpt150")
    assert sorted(path.name for path in target.iterdir()) == sorted(ADAPTER_FILES)
    export_adapter(checkpoint, target)  # again: the folder is made new


def test_a_checkpoint_without_weights_is_refused(tmp_path):
    checkpoint = tmp_path / "checkpoint-150"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        export_adapter(checkpoint, tmp_path / "out")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A smoke run of four steps with a checkpoint at each step, and its manifest."""
    adapters = tmp_path_factory.mktemp("adapters")
    settings = TrainSettings("tax-booking", 0, smoke=True, max_steps=4, adapters_dir=adapters)
    run(settings)
    return adapters


@pytest.fixture
def adapters(trained, tmp_path):
    copy = tmp_path / "adapters"
    shutil.copytree(trained, copy)
    return copy


def choose_in(adapters, tmp_path, **changes):
    options = dict(
        smoke=True, dry_run=True, adapters_dir=adapters, steps=[2, 3, 4], workers=2,
        servers=1,
    )  # fmt: skip
    return select_checkpoint(load_registry(), None, 0, tmp_path / "eval", **{**options, **changes})


def manifest_of(adapters):
    return json.loads((adapters / "tax-booking" / "tax-booking-smoke-0.json").read_text("utf-8"))


def test_the_run_leaves_no_selection_until_one_is_made(adapters):
    assert manifest_of(adapters)["selection"] is None


def test_every_checkpoint_is_scored_on_dev_real_with_no_confidence_limit(adapters, tmp_path):
    selection = choose_in(adapters, tmp_path)
    assert [c["step"] for c in selection["candidates"]] == [2, 3, 4]
    for step in (2, 3, 4):
        path = results_file(tmp_path / "eval", "tax-booking", f"smoke-sft-0-ckpt{step}", "dev_real")
        results = json.loads(path.read_text(encoding="utf-8"))
        assert results["split"] == "dev_real"
        assert results["manifest"]["threshold"] == 0.0
        assert results["manifest"]["adapter_hash"] == hash_directory(
            adapters / "tax-booking" / f"smoke-0-ckpt{step}"
        )
    assert selection["split"] == "dev_real" and selection["method"] == METHOD
    assert selection["file"] == "dev_real.jsonl"
    assert selection["sha256"] == hash_file(split_path("tax-booking", "dev_real"))
    assert selection["examples"] > 0 and selection["confidence_limit"] == 0.0


def test_the_choice_and_the_scores_are_written_into_the_manifest(adapters, tmp_path):
    selection = choose_in(adapters, tmp_path)
    manifest = manifest_of(adapters)
    assert manifest["selection"] == selection
    for entry in selection["candidates"]:
        assert set(entry) == {
            "step",
            "system",
            "critical_error_rate",
            "document_accuracy",
            "automation_rate",
            "adapter_sha256",
        }
    assert selection["chosen_step"] == choose(selection["candidates"])["step"]
    assert manifest["steps"] == 4  # the number of steps trained does not change


def test_the_chosen_checkpoint_takes_the_place_of_the_adapter(adapters, tmp_path, monkeypatch):
    # The first candidate looks best, the others worse.
    rates = iter([0.0, 0.5, 0.5])
    monkeypatch.setattr(
        selection_module,
        "scores",
        lambda results: {
            "critical_error_rate": next(rates),
            "document_accuracy": 1.0,
            "automation_rate": 1.0,
        },
    )
    selection = choose_in(adapters, tmp_path)
    assert selection["chosen_step"] == 2
    adapter = adapters / "tax-booking" / "smoke-0"
    checkpoint = adapters / "tax-booking" / "smoke-0-checkpoints" / "checkpoint-2"
    assert sorted(path.name for path in adapter.iterdir()) == sorted(ADAPTER_FILES)
    for name in ADAPTER_FILES:
        assert (adapter / name).read_bytes() == (checkpoint / name).read_bytes()
    assert manifest_of(adapters)["adapter_sha256"] == hash_directory(adapter)
    by_step = {c["step"]: c for c in selection["candidates"]}
    assert by_step[2]["adapter_sha256"] == hash_directory(adapter)


def test_the_checkpoints_stay_where_they_were(adapters, tmp_path):
    folder = adapters / "tax-booking" / "smoke-0-checkpoints"
    before = sorted(path.name for path in folder.rglob("*"))
    choose_in(adapters, tmp_path)
    assert sorted(path.name for path in folder.rglob("*")) == before


def test_a_second_selection_is_refused(adapters, tmp_path):
    choose_in(adapters, tmp_path)
    with pytest.raises(ValueError, match="has a selection already"):
        choose_in(adapters, tmp_path)


def test_a_run_without_a_manifest_is_refused(adapters, tmp_path):
    (adapters / "tax-booking" / "tax-booking-smoke-0.json").unlink()
    with pytest.raises(ValueError, match="train the run first"):
        choose_in(adapters, tmp_path)


def test_a_step_without_a_checkpoint_is_refused_before_anything_is_changed(adapters, tmp_path):
    adapter = adapters / "tax-booking" / "smoke-0"
    before = hash_directory(adapter)
    with pytest.raises(ValueError, match="no checkpoint for step 150"):
        choose_in(adapters, tmp_path, steps=[2, 150])
    assert hash_directory(adapter) == before and manifest_of(adapters)["selection"] is None
    assert not (tmp_path / "eval").exists()


def test_the_command_prints_the_scores_and_marks_the_chosen_step(adapters, tmp_path, capsys):
    arguments = ["--smoke", "--dry-run", "--steps", "3", "4", "--adapters-dir", str(adapters)]
    assert main([*arguments, "--out-dir", str(tmp_path / "eval")]) == 0
    out = capsys.readouterr().out
    assert "lowest critical error rate" in out
    assert out.count("<- chosen") == 1
    assert "step    3" in out and "step    4" in out


def test_the_command_needs_a_model_unless_it_is_a_smoke_run():
    with pytest.raises(SystemExit) as stop:
        main([])
    assert stop.value.code == 2


def test_the_command_reports_a_problem_as_an_error(tmp_path):
    with pytest.raises(SystemExit) as stop:
        main(["--smoke", "--dry-run", "--adapters-dir", str(tmp_path), "--steps", "2"])
    assert "train the run first" in str(stop.value)

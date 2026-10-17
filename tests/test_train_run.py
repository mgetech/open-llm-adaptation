"""The parts of a training run that need no GPU library: names, refusals and folders."""

import pytest
import yaml

from adapt.models import MODELS_FILE, Registry, load_registry
from adapt.train.run import (
    TrainSettings,
    find_checkpoint,
    main,
    prepare_output,
    run,
    run_name,
    trainable_entry,
)


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


def test_the_run_name_has_the_model_and_the_seed():
    assert run_name(TrainSettings("tax-booking", 2, model="qwen3-4b")) == "qwen3-4b-2"


def test_a_short_check_has_its_own_name_so_it_never_replaces_a_full_run():
    settings = TrainSettings("tax-booking", 0, model="qwen3-4b", max_steps=3)
    assert run_name(settings) == "qwen3-4b-0-steps3"


def test_a_smoke_run_has_its_own_name():
    assert run_name(TrainSettings("tax-booking", 1, smoke=True)) == "smoke-1"


def test_a_candidate_may_be_fine_tuned(registry):
    assert trainable_entry(registry, "qwen3-1.7b").hf_id == "Qwen/Qwen3-1.7B"


def test_the_reference_model_is_not_fine_tuned(registry):
    with pytest.raises(ValueError, match="qwen3-8b is a reference model"):
        trainable_entry(registry, "qwen3-8b")


def test_an_unknown_model_is_refused(registry):
    with pytest.raises(ValueError, match="unknown model 'gpt-9'.*qwen3-1.7b"):
        trainable_entry(registry, "gpt-9")


def test_a_model_that_fails_a_gate_is_refused_before_any_work(tmp_path):
    data = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
    data["models"]["qwen3-4b"]["licence"] = "gpl-3.0"
    changed = Registry.model_validate(data)
    settings = TrainSettings("tax-booking", 0, model="qwen3-4b", adapters_dir=tmp_path)
    with pytest.raises(ValueError, match="qwen3-4b fails the model gates: licence"):
        run(settings, registry=changed)
    assert list(tmp_path.iterdir()) == []


def test_the_command_needs_a_model_unless_it_is_a_smoke_run():
    with pytest.raises(SystemExit) as stop:
        main([])
    assert stop.value.code == 2


def test_the_command_refuses_the_reference_model(tmp_path):
    with pytest.raises(SystemExit) as stop:
        main(["--model", "qwen3-8b", "--adapters-dir", str(tmp_path)])
    assert "only candidates are fine-tuned" in str(stop.value)


def test_the_latest_checkpoint_is_the_one_with_the_highest_step(tmp_path):
    for name in ["checkpoint-2", "checkpoint-10", "checkpoint-100", "checkpoint-x"]:
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-500").write_text("a file, not a checkpoint")
    assert find_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_no_checkpoint_gives_none(tmp_path):
    assert find_checkpoint(tmp_path) is None
    assert find_checkpoint(tmp_path / "missing") is None


@pytest.fixture
def folders(tmp_path):
    adapter, checkpoints = tmp_path / "adapter", tmp_path / "checkpoints"
    return adapter, checkpoints


def test_a_new_run_may_use_empty_folders(folders):
    prepare_output(*folders, resume=False, smoke=False)


def test_a_new_run_does_not_replace_an_adapter(folders):
    adapter, checkpoints = folders
    adapter.mkdir()
    with pytest.raises(ValueError, match="Use --resume"):
        prepare_output(adapter, checkpoints, resume=False, smoke=False)
    assert adapter.exists()


def test_a_new_run_does_not_mix_with_old_checkpoints(folders):
    adapter, checkpoints = folders
    (checkpoints / "checkpoint-100").mkdir(parents=True)
    with pytest.raises(ValueError, match="Use --resume"):
        prepare_output(adapter, checkpoints, resume=False, smoke=False)
    assert (checkpoints / "checkpoint-100").exists()


def test_a_resumed_run_keeps_the_files(folders):
    adapter, checkpoints = folders
    adapter.mkdir()
    (checkpoints / "checkpoint-100").mkdir(parents=True)
    prepare_output(adapter, checkpoints, resume=True, smoke=False)
    assert adapter.exists() and (checkpoints / "checkpoint-100").exists()


def test_a_smoke_run_replaces_the_files_of_the_last_smoke_run(folders):
    adapter, checkpoints = folders
    adapter.mkdir()
    (checkpoints / "checkpoint-1").mkdir(parents=True)
    prepare_output(adapter, checkpoints, resume=False, smoke=True)
    assert not adapter.exists() and not checkpoints.exists()

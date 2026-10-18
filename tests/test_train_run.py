"""The parts of a training run that need no GPU library: names, refusals and folders."""

import shutil
import subprocess

import pytest
import yaml

from adapt.data.rounds import current_round
from adapt.eval.runner import ROOT
from adapt.models import MODELS_FILE, Registry, load_registry
from adapt.train.config import load_train_config
from adapt.train.run import (
    TrainSettings,
    accumulation_steps,
    check_processes,
    find_checkpoint,
    main,
    prepare_output,
    process_ranks,
    run,
    run_name,
    trainable_entry,
)


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


def test_the_run_name_has_the_model_and_the_seed():
    assert run_name(TrainSettings("tax-booking", 2, model="qwen3-4b", data_round=1)) == "qwen3-4b-2"


def test_a_later_round_adds_its_number_after_the_model():
    settings = TrainSettings("tax-booking", 2, model="qwen3-4b", data_round=2)
    assert run_name(settings) == "qwen3-4b-r2-2"
    short = TrainSettings("tax-booking", 0, model="qwen3-4b", max_steps=3, data_round=2)
    assert run_name(short) == "qwen3-4b-r2-0-steps3-gpus1"


def test_a_run_without_a_round_uses_the_current_one():
    current = current_round()
    assert current >= 2
    assert run_name(TrainSettings("tax-booking", 0, model="qwen3-4b")) == (f"qwen3-4b-r{current}-0")


def test_a_round_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="does not exist"):
        run_name(TrainSettings("tax-booking", 0, model="qwen3-4b", data_round=current_round() + 1))


def test_a_short_check_has_its_own_name_so_it_never_replaces_a_full_run():
    settings = TrainSettings("tax-booking", 0, model="qwen3-4b", max_steps=3, data_round=1)
    assert run_name(settings) == "qwen3-4b-0-steps3-gpus1"


def test_a_short_run_on_one_gpu_does_not_replace_the_same_run_on_two():
    one = TrainSettings("tax-booking", 0, model="qwen3-1.7b", max_steps=50, data_round=1)
    two = TrainSettings(
        "tax-booking", 0, model="qwen3-1.7b", max_steps=50, num_gpus=2, data_round=1
    )
    assert run_name(one) == "qwen3-1.7b-0-steps50-gpus1"
    assert run_name(two) == "qwen3-1.7b-0-steps50-gpus2"


def test_a_full_run_has_the_same_name_on_any_number_of_gpus():
    two = TrainSettings("tax-booking", 1, model="qwen3-4b", num_gpus=2, data_round=1)
    assert run_name(two) == "qwen3-4b-1"


def test_a_smoke_run_has_its_own_name():
    assert run_name(TrainSettings("tax-booking", 1, smoke=True)) == "smoke-1"


def test_a_candidate_may_be_fine_tuned(registry):
    assert trainable_entry(registry, "qwen3-1.7b").hf_id == "Qwen/Qwen3-1.7B"


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


def test_a_run_started_without_the_launcher_is_one_process(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert process_ranks() == (0, 1)


def test_the_ranks_come_from_the_launcher(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert process_ranks() == (1, 2)


def test_the_number_of_processes_must_match_the_number_of_gpus(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    check_processes(2)
    with pytest.raises(ValueError, match="2 processes were started for 1 GPUs.*NUM_GPUS"):
        check_processes(1)


def test_two_gpus_asked_for_without_the_launcher_stop_the_run_before_any_work(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(SystemExit) as stop:
        main(["--model", "qwen3-4b", "--num-gpus", "2", "--adapters-dir", str(tmp_path)])
    assert "1 processes were started for 2 GPUs" in str(stop.value)
    assert list(tmp_path.iterdir()) == []


def test_no_gpu_is_not_a_valid_number_of_gpus(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match="at least 1"):
        check_processes(0)


@pytest.mark.parametrize("num_gpus", [1, 2, 4])
def test_the_effective_batch_size_is_the_same_on_any_number_of_gpus(num_gpus):
    optimiser = load_train_config().optimiser
    accumulation = accumulation_steps(optimiser.gradient_accumulation_steps, num_gpus)
    assert optimiser.batch_size * accumulation * num_gpus == 16


@pytest.mark.parametrize("num_gpus", [3, 8])
def test_a_number_of_gpus_that_would_change_the_batch_size_is_refused(num_gpus):
    total = load_train_config().optimiser.gradient_accumulation_steps
    with pytest.raises(ValueError, match="effective batch size"):
        accumulation_steps(total, num_gpus)


needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def planned(*variables: str) -> str:
    """What `make train` would run, without running it."""
    result = subprocess.run(
        ["make", "--dry-run", "train", *variables],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return " ".join(result.stdout.replace("\\\n", " ").split())


@needs_make
def test_two_gpus_are_started_with_the_launcher_and_two_processes():
    command = planned("MODEL=qwen3-4b", "NUM_GPUS=2")
    assert (
        "accelerate launch --multi_gpu --num_processes 2 -m adapt.train "
        "--platform tax-booking --seed 0 --num-gpus 2 --model qwen3-4b" in command
    )


@needs_make
def test_one_gpu_is_the_default_and_needs_no_launcher():
    command = planned("MODEL=qwen3-4b")
    assert "python -m adapt.train --platform tax-booking --seed 0 --num-gpus 1" in command
    assert "accelerate" not in command


@needs_make
def test_a_short_run_passes_its_steps_and_gpus_on():
    command = planned("MODEL=qwen3-1.7b", "NUM_GPUS=2", "MAX_STEPS=50")
    assert "--num_processes 2" in command and "--num-gpus 2" in command
    assert "--max-steps 50" in command

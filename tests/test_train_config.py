import pytest
import yaml
from pydantic import ValidationError

from adapt.train.config import TRAIN_FILE, TrainConfig, load_train_config


@pytest.fixture(scope="module")
def config() -> TrainConfig:
    return load_train_config()


def test_the_quantisation_is_nf4_with_double_quantisation_and_fp16_compute(config):
    quantisation = config.quantisation
    assert quantisation.quant_type == "nf4"
    assert quantisation.double_quant is True
    assert quantisation.compute_dtype == "float16"  # a T4 has no bf16


def test_the_lora_settings(config):
    lora = config.lora
    assert (lora.r, lora.alpha, lora.dropout) == (16, 32, 0.05)
    assert lora.target_modules == "all-linear"


def test_the_optimiser_settings(config):
    optimiser = config.optimiser
    assert optimiser.learning_rate == 2e-4
    assert optimiser.lr_scheduler == "cosine"
    assert optimiser.warmup_ratio == 0.03
    assert optimiser.epochs == 1
    assert optimiser.optim == "paged_adamw_8bit"
    assert optimiser.batch_size * optimiser.gradient_accumulation_steps == 16
    assert (optimiser.batch_size, optimiser.gradient_accumulation_steps) == (4, 4)
    assert optimiser.gradient_checkpointing is True


def test_the_length_limit_and_the_checkpoint_interval(config):
    assert config.data.max_length == 1280
    assert config.checkpoint.save_steps == 75


def test_a_full_run_leaves_the_checkpoints_that_are_scored_afterwards(config):
    steps = config.optimiser.epochs * 6000 // 16
    assert steps == 375
    saved = range(config.checkpoint.save_steps, steps + 1, config.checkpoint.save_steps)
    assert list(saved) == [75, 150, 225, 300, 375]
    assert config.selection.steps == [150, 225, 300, 375]
    assert set(config.selection.steps) <= set(saved)


def test_there_is_no_loss_based_early_stopping(config):
    assert not hasattr(config, "early_stopping")
    assert not hasattr(config.checkpoint, "keep")  # every checkpoint is kept


def test_a_step_without_a_checkpoint_cannot_be_selected(tmp_path):
    data = yaml.safe_load(TRAIN_FILE.read_text(encoding="utf-8"))
    data["selection"]["steps"] = [150, 200]
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError, match="no checkpoint is saved at step 200"):
        load_train_config(path)


def test_a_smoke_run_is_two_steps(config):
    assert config.smoke.steps == 2


def test_the_tiny_model_fits_the_tiny_attention_layout(config):
    smoke = config.smoke
    assert smoke.heads % smoke.kv_heads == 0
    assert smoke.max_length >= config.data.max_length


def test_a_setting_that_is_not_known_is_refused(tmp_path):
    data = yaml.safe_load(TRAIN_FILE.read_text(encoding="utf-8"))
    data["lora"]["rank"] = 8
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError, match="rank"):
        load_train_config(path)


def test_a_value_out_of_range_is_refused(tmp_path):
    data = yaml.safe_load(TRAIN_FILE.read_text(encoding="utf-8"))
    data["lora"]["dropout"] = 1.5
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError, match="dropout"):
        load_train_config(path)

"""The training settings of `configs/train.yaml`."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

TRAIN_FILE = Path(__file__).resolve().parents[3] / "configs" / "train.yaml"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Quantisation(_Model):
    quant_type: Literal["nf4", "fp4"]
    double_quant: bool
    compute_dtype: Literal["float16", "bfloat16"]


class Lora(_Model):
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: str


class Optimiser(_Model):
    learning_rate: float = Field(gt=0)
    lr_scheduler: str
    warmup_ratio: float = Field(ge=0, lt=1)
    epochs: int = Field(gt=0)
    optim: str
    batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    gradient_checkpointing: bool


class Data(_Model):
    max_length: int = Field(gt=0)


class Checkpoint(_Model):
    save_steps: int = Field(gt=0)
    keep: int = Field(gt=0)


class Smoke(_Model):
    steps: int = Field(gt=0)
    examples: int = Field(gt=0)
    tokenizer_examples: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_length: int = Field(gt=0)
    save_steps: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    layers: int = Field(gt=0)
    heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)


class TrainConfig(_Model):
    quantisation: Quantisation
    lora: Lora
    optimiser: Optimiser
    data: Data
    checkpoint: Checkpoint
    logging_steps: int = Field(gt=0)
    smoke: Smoke


def load_train_config(path: Path = TRAIN_FILE) -> TrainConfig:
    with path.open(encoding="utf-8") as file:
        return TrainConfig.model_validate(yaml.safe_load(file))

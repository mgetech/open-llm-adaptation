"""Training on the CPU with the tiny model. These tests need the `gpu` dependency group."""

import hashlib
import json
import os

import pytest

from adapt.config import load_platform
from adapt.eval.runner import hash_directory, hash_file, split_path
from adapt.prompting import build_messages, system_prompt
from adapt.train.data import (
    IGNORE,
    answer_text,
    encode_example,
    encode_records,
    load_train_records,
    render_prompt,
)
from adapt.train.run import TrainSettings, lora_to_float32, main, run
from adapt.train.smoke import tiny_tokenizer
from chat_template import TEMPLATE, render_chat

for library in ("torch", "transformers", "trl", "peft", "datasets", "tokenizers"):
    pytest.importorskip(library)

EMPTY_THINKING = "<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.fixture(scope="module")
def records():
    return load_train_records("tax-booking")


@pytest.fixture(scope="module")
def system(platform):
    return system_prompt("tax-booking", "SKR03", platform.rules)


@pytest.fixture(scope="module")
def tokenizer(records, system):
    texts = [r["document_text"] + "\n" + json.dumps(r["target"]) for r in records[:100]]
    return tiny_tokenizer([*texts, system], vocab_size=1000)


@pytest.fixture(scope="module")
def qwen_tokenizer(records, system):
    """The same tiny tokenizer, with the chat template of the Qwen3 models."""
    texts = [r["document_text"] + "\n" + json.dumps(r["target"]) for r in records[:100]]
    qwen = tiny_tokenizer([*texts, system], vocab_size=1000)
    qwen.chat_template = TEMPLATE.read_text(encoding="utf-8")
    return qwen


def decode_labelled(tokenizer, example):
    ids = [
        i
        for i, label in zip(example["input_ids"], example["labels"], strict=True)
        if label != IGNORE
    ]
    return tokenizer.decode(ids)


def test_the_smoke_template_renders_like_the_qwen_template(tokenizer, records, system):
    for record in records[:5]:
        for thinking in (False, True):
            expected = render_chat(
                build_messages(system, record["document_text"]),
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            assert render_prompt(tokenizer, system, record["document_text"], thinking) == expected


def test_the_prompt_is_what_serving_builds_with_the_real_chat_template(
    qwen_tokenizer, records, system
):
    for record in records[:20]:
        prompt = render_prompt(qwen_tokenizer, system, record["document_text"], False)
        expected = render_chat(
            build_messages(system, record["document_text"]),
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert prompt == expected
        assert prompt.endswith("<|im_start|>assistant\n" + EMPTY_THINKING)


def test_the_answer_is_the_target_as_json_and_the_end_token(tokenizer, records):
    record = records[0]
    assert answer_text(tokenizer, record) == json.dumps(record["target"]) + "<|im_end|>"
    assert tokenizer.eos_token == "<|im_end|>"


def test_only_the_answer_and_the_end_token_are_learned(tokenizer, records, system):
    for record in records[:10]:
        example = encode_example(tokenizer, system, record, enable_thinking=False, max_length=4096)
        assert len(example["input_ids"]) == len(example["labels"])
        assert decode_labelled(tokenizer, example) == answer_text(tokenizer, record)
        # The labels are the answer ids at the end, and nothing before.
        first = next(i for i, label in enumerate(example["labels"]) if label != IGNORE)
        assert set(example["labels"][:first]) == {IGNORE}
        assert example["labels"][first:] == example["input_ids"][first:]
        prompt = render_prompt(tokenizer, system, record["document_text"], False)
        assert tokenizer.decode(example["input_ids"][:first]) == prompt


def test_the_learned_answer_parses_back_to_the_target(tokenizer, records, system):
    record = records[3]
    example = encode_example(tokenizer, system, record, enable_thinking=False, max_length=4096)
    text = decode_labelled(tokenizer, example).removesuffix("<|im_end|>")
    assert json.loads(text) == record["target"]


def test_the_last_learned_token_is_the_end_token(tokenizer, records, system):
    example = encode_example(tokenizer, system, records[0], enable_thinking=False, max_length=4096)
    assert example["labels"][-1] == tokenizer.eos_token_id


def test_an_example_over_the_limit_stops_the_run(tokenizer, records, system):
    with pytest.raises(ValueError, match=r"train-00000 has \d+ tokens, the limit is 50"):
        encode_example(tokenizer, system, records[0], enable_thinking=False, max_length=50)


def test_thinking_is_disabled_for_the_models_of_the_registry(tokenizer, records, system):
    on = encode_example(tokenizer, system, records[0], enable_thinking=True, max_length=4096)
    off = encode_example(tokenizer, system, records[0], enable_thinking=False, max_length=4096)
    # The disabled case has the empty thinking block between the prompt and the answer.
    assert len(off["input_ids"]) > len(on["input_ids"])
    assert EMPTY_THINKING in tokenizer.decode(off["input_ids"])
    assert EMPTY_THINKING not in tokenizer.decode(on["input_ids"])


def test_each_record_is_encoded_in_order(tokenizer, platform, records):
    examples = encode_records(
        tokenizer,
        "tax-booking",
        platform.rules,
        records[:6],
        enable_thinking=False,
        max_length=4096,
    )
    assert len(examples) == 6
    for example, record in zip(examples, records[:6], strict=True):
        assert decode_labelled(tokenizer, example) == answer_text(tokenizer, record)


def test_trainable_weights_go_back_to_fp32_and_frozen_ones_stay():
    import torch

    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    model.weight.requires_grad_(True)
    model.bias.requires_grad_(False)
    lora_to_float32(model)
    assert model.weight.dtype == torch.float32
    assert model.bias.dtype == torch.bfloat16


def test_a_real_model_needs_a_gpu(tmp_path, monkeypatch):
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    with pytest.raises(SystemExit) as stop:
        main(["--model", "qwen3-1.7b", "--adapters-dir", str(tmp_path)])
    assert "no GPU found" in str(stop.value)
    assert not (tmp_path / "tax-booking" / "qwen3-1.7b-0").exists()


def test_a_real_run_uses_the_first_gpu_only_unless_told_otherwise(tmp_path, monkeypatch):
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    # Set, then removed, so that monkeypatch also removes what the command sets.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(SystemExit):
        main(["--model", "qwen3-1.7b", "--adapters-dir", str(tmp_path)])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_a_run_on_two_gpus_makes_both_visible(tmp_path, monkeypatch):
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("WORLD_SIZE", "2")
    arguments = ["--model", "qwen3-1.7b", "--num-gpus", "2", "--adapters-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        main(arguments)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_a_choice_of_gpu_is_kept(tmp_path, monkeypatch):
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(SystemExit):
        main(["--model", "qwen3-1.7b", "--adapters-dir", str(tmp_path)])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    adapters = tmp_path_factory.mktemp("adapters")
    return adapters, run(TrainSettings("tax-booking", 0, smoke=True, adapters_dir=adapters))


def test_a_smoke_run_writes_an_adapter(smoke):
    adapters, manifest = smoke
    adapter = adapters / "tax-booking" / "smoke-0"
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert (adapter / "adapter_model.safetensors").stat().st_size > 0
    assert (config["r"], config["lora_alpha"], config["lora_dropout"]) == (16, 32, 0.05)
    assert manifest["adapter_sha256"] == hash_directory(adapter)


def test_a_smoke_run_trains_two_steps_and_keeps_its_manifest_out_of_results(smoke):
    adapters, manifest = smoke
    path = adapters / "tax-booking" / "tax-booking-smoke-0.json"
    assert manifest["manifest_file"] == path.as_posix()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {k: v for k, v in manifest.items() if k != "manifest_file"}
    assert saved["steps"] == 2 and saved["resumed_from_step"] is None
    assert 0 < saved["train_loss"] < 20


def test_the_manifest_records_what_was_run(smoke):
    _, manifest = smoke
    train_file = split_path("tax-booking", "train")
    prompt = system_prompt("tax-booking", "SKR03", load_platform("tax-booking").rules)
    assert manifest["run_id"] == "tax-booking-smoke-0"
    assert manifest["seed"] == 0 and manifest["smoke"] is True and manifest["model"] is None
    assert manifest["config"]["lora"]["r"] == 16
    assert manifest["data"]["train_sha256"] == hash_file(train_file)
    assert manifest["data"]["examples"] == 16
    assert manifest["data"]["longest_example_tokens"] <= manifest["data"]["max_length"]
    assert manifest["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert isinstance(manifest["git_dirty"], bool | type(None))
    assert manifest["gpu"] is None and manifest["gpu_count"] == 0 and manifest["gpu_hours"] == 0
    assert manifest["wall_seconds"] > 0
    assert manifest["num_gpus"] == 1 and manifest["peak_gpu_memory_gib"] is None
    assert manifest["per_device_batch_size"] == 2 and manifest["gradient_accumulation_steps"] == 1
    assert manifest["effective_batch_size"] == 2
    assert {"python", "torch", "transformers", "trl", "peft"} <= set(manifest["versions"])
    assert manifest["trainable_parameters"] > 0


def test_a_smoke_run_leaves_a_checkpoint(smoke):
    adapters, _ = smoke
    folder = adapters / "tax-booking" / "smoke-0-checkpoints"
    assert (folder / "checkpoint-2").is_dir()


def test_the_same_seed_gives_the_same_training(smoke, tmp_path):
    _, first = smoke
    second = run(TrainSettings("tax-booking", 0, smoke=True, adapters_dir=tmp_path))
    assert second["train_loss"] == first["train_loss"]
    assert second["adapter_sha256"] == first["adapter_sha256"]


def test_a_run_continues_from_its_last_checkpoint(tmp_path):
    first = run(TrainSettings("tax-booking", 0, smoke=True, max_steps=1, adapters_dir=tmp_path))
    assert first["steps"] == 1
    second = run(
        TrainSettings("tax-booking", 0, smoke=True, max_steps=2, resume=True, adapters_dir=tmp_path)
    )
    assert second["resumed_from_step"] == 1 and second["steps"] == 2
    assert (tmp_path / "tax-booking" / "smoke-0-checkpoints" / "checkpoint-2").is_dir()

"""The model card states facts that also live in configs and run manifests. They must agree."""

import hashlib
import json

import pytest
import yaml

from adapt.eval.runner import ROOT
from adapt.models import load_registry
from adapt.serve import adapter_path, load_serve_config

CARD = ROOT / "docs" / "model-card.md"
LICENSE_NOTE = ROOT / "docs" / "adapter-license.txt"
MANIFEST = ROOT / "results" / "round3" / "runs" / "tax-booking-qwen3-4b-r3-0.json"


@pytest.fixture(scope="module")
def card():
    return CARD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def front_matter(card):
    _, header, _ = card.split("---\n", 2)
    return yaml.safe_load(header)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_header_names_the_base_model(front_matter):
    entry = load_registry().models[load_serve_config().model]
    assert front_matter["base_model"] == entry.hf_id
    assert front_matter["library_name"] == "peft"


def test_the_adapter_is_non_commercial_and_the_base_model_keeps_its_licence(card, front_matter):
    entry = load_registry().models[load_serve_config().model]
    assert front_matter["license"] == "cc-by-nc-4.0"
    assert entry.licence == "apache-2.0" and "licensed under Apache 2.0" in card
    assert "CC BY-NC 4.0" in card and "Not for commercial use." in card
    assert "For research and demonstration only." in card


def test_the_access_box_shows_the_terms_and_asks_for_a_tick(front_matter):
    assert "Not for commercial use." in front_matter["extra_gated_prompt"]
    assert front_matter["extra_gated_fields"] == {
        "I will not use this adapter commercially": "checkbox"
    }


def test_the_license_file_for_the_hub_says_the_same():
    text = LICENSE_NOTE.read_text(encoding="utf-8")
    assert "CC BY-NC 4.0" in text and "Not for commercial use." in text
    assert "creativecommons.org/licenses/by-nc/4.0" in text
    assert "Apache 2.0" in text


def test_the_card_gives_the_pinned_revision_and_the_serving_flags(card):
    config = load_serve_config()
    entry = load_registry().models[config.model]
    assert entry.revision in card
    assert f"--max-model-len {config.max_model_len}" in card
    assert f"--max-lora-rank {config.lora.max_lora_rank}" in card


def test_the_card_gives_the_hash_and_the_threshold_of_the_serve_file(card):
    (adapter,) = load_serve_config().adapters
    assert adapter.sha256 in card
    assert f"{adapter.threshold:.4f}" in card


def test_the_card_agrees_with_the_training_manifest(card, manifest):
    assert manifest["adapter_sha256"] in card
    assert manifest["prompt_sha256"] in card
    assert manifest["model_revision"] in card
    assert f"{manifest['trainable_parameters']:,}" in card
    assert f"{manifest['data']['examples']:,}" in card
    assert f"{int(manifest['wall_seconds']):,}" in card
    assert f"{manifest['gpu_hours']:.1f} GPU hours" in card
    assert f"step {manifest['selection']['chosen_step']}" in card
    assert f"({manifest['steps']} steps)" in card
    assert manifest["config"]["lora"]["r"] == 16 and "rank 16" in card


def test_the_file_hashes_of_the_card_are_those_of_the_adapter_when_it_is_on_disk(card):
    (adapter,) = load_serve_config().adapters
    folder = adapter_path(adapter)
    if not (folder / "adapter_model.safetensors").is_file():
        pytest.skip("the adapter is not on this machine")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        text = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        assert f"`{name}` | `{text}`" in card


def test_the_card_says_the_held_out_run_failed_and_the_other_numbers_are_not_held_out(card):
    assert "did not pass the gate" in card
    assert "Validation, not a test" in card
    assert "Post-hoc, on previously used sets, not held out" in card

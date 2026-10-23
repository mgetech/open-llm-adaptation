"""The sizing note states numbers that also live in run manifests and logs. They must agree."""

import json
import re

import pytest

from adapt.eval.runner import ROOT

NOTE = (ROOT / "docs" / "gpu-sizing.md").read_text(encoding="utf-8")
LOG = (ROOT / "results" / "serving" / "vllm-memory.txt").read_text(encoding="utf-8")
RUNS = ROOT / "results" / "round1" / "runs"
AZURE_DEV = ROOT / "results" / "azure" / "eval" / "tax-booking" / "qwen3-4b-sft-r3-0-azure"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scaling():
    one = load(RUNS / "tax-booking-qwen3-1.7b-0-steps50-gpus1.json")
    two = load(RUNS / "tax-booking-qwen3-1.7b-0-steps50-gpus2.json")
    assert (one["num_gpus"], two["num_gpus"]) == (1, 2)
    return one, two


def test_the_training_table_is_the_two_short_runs(scaling):
    one, two = scaling
    samples = one["steps"] * one["effective_batch_size"]
    assert one["effective_batch_size"] == two["effective_batch_size"] and one["steps"] == 50
    for text in (
        f"{one['wall_seconds']:.1f} s",
        f"{two['wall_seconds']:.1f} s",
        f"{samples / one['wall_seconds']:.2f}",
        f"{samples / two['wall_seconds']:.2f}",
        f"{one['gpu_hours']:.3f}",
        f"{two['gpu_hours']:.3f}",
        f"{one['peak_gpu_memory_gib']:.2f} GiB",
        f"{two['peak_gpu_memory_gib']:.2f} GiB",
    ):
        assert text in NOTE, text


def test_the_speed_up_the_efficiency_and_the_epoch_times_follow_from_the_runs(scaling):
    one, two = scaling
    speedup = one["wall_seconds"] / two["wall_seconds"]
    assert f"{speedup:.2f} times" in NOTE and f"{speedup / 2:.0%}" in NOTE
    for run in (one, two):
        minutes = run["wall_seconds"] / run["steps"] * 375 / 60
        assert f"{minutes:.0f} min" in NOTE


def test_the_shipped_run_is_the_one_of_the_manifest():
    run = load(ROOT / "results" / "round3" / "runs" / "tax-booking-qwen3-4b-r3-0.json")
    assert run["num_gpus"] == 2 and run["steps"] == 375
    assert f"{run['wall_seconds']:,.0f} seconds ({run['gpu_hours']:.1f} GPU hours)" in NOTE
    assert f"{run['peak_gpu_memory_gib']:.1f} GiB per GPU" in NOTE


def test_the_calculated_kv_cache_matches_the_tokens_vllm_reports():
    layers, kv_heads, head_size, fp16 = 36, 8, 128, 2
    per_token = 2 * layers * kv_heads * head_size * fp16
    tokens = int(re.search(r"GPU KV cache size: ([\d,]+) tokens", LOG).group(1).replace(",", ""))
    assert per_token == 147_456 and "147,456 bytes per token" in NOTE
    assert f"{tokens:,} tokens are {per_token * tokens / 2**30:.2f} GiB" in NOTE
    assert f"{tokens / 4096:.2f} requests of 4,096 tokens" in NOTE
    assert "Maximum concurrency for 4,096 tokens per request: 6.05x" in LOG
    assert f"at least {tokens // 1280} documents" in NOTE


def test_the_weights_and_the_memory_lines_are_the_ones_of_the_log():
    assert f"{8_044_936_192 / 2**30:.2f} GiB" in NOTE
    for text in ("Model loading took 7.7 GiB", "Available KV cache memory: 3.4 GiB"):
        assert text in LOG
    assert "7.7 GiB" in NOTE and "3.4 GiB" in NOTE
    assert "(0.9, 13.11 GiB)" in LOG and "1.65 GiB for peak activation" in LOG
    assert "0.45 GiB for CUDAGraph" in LOG and "(14.46/14.56 GiB)" in LOG
    assert "13.11 GiB" in NOTE and "14.56 GiB" in NOTE and "1.65 GiB and 0.45 GiB" in NOTE


def test_the_adapter_size_is_from_the_manifest_and_the_file_size():
    run = load(ROOT / "results" / "round3" / "runs" / "tax-booking-qwen3-4b-r3-0.json")
    parameters = run["trainable_parameters"]
    assert f"{parameters:,} parameters" in NOTE
    assert f"{parameters * 2 / 2**20:.0f} MiB in fp16" in NOTE
    assert "126 MiB in 32 bits" in NOTE
    file = ROOT / "adapters" / "tax-booking" / "qwen3-4b-r3-0" / "adapter_model.safetensors"
    if file.is_file():
        assert f"{file.stat().st_size / 2**20:.0f} MiB in 32 bits" in NOTE


def test_the_capacity_is_the_azure_run_of_dev_and_the_table_follows_from_it():
    saved = load(AZURE_DEV / "dev.json")
    performance = saved["performance"]
    rate = performance["documents_per_second"]
    assert performance["workers"] == 8 and saved["manifest"]["documents"] == 300
    for text in (
        f"{performance['wall_seconds']:.1f} seconds",
        f"{rate:.3f} documents per second",
        f"{performance['latency_p50_seconds']:.1f} seconds",
        f"{performance['latency_p95_seconds']:.1f} seconds",
        f"about {round(rate * 86400, -2):,.0f} documents",
    ):
        assert text in NOTE, text
    for documents in (5_000, 20_000, 100_000):
        busy = documents / (rate * 86400)
        assert f"| {documents:,} | {busy:.0%} | {-(-busy // 1):.0f} |" in NOTE


def test_the_smoke_test_latencies_and_the_decode_speed_are_those_of_the_files():
    documents = load(ROOT / "results" / "serving" / "smoke-test.json")["documents"]
    seconds = [item["latency_ms"] / 1000 for item in documents]
    assert (
        f"{min(seconds):.1f} to {max(seconds):.1f} seconds per document for {len(seconds)}" in NOTE
    )
    assert "Avg generation throughput: 11.6 tokens/s" in LOG and "11.6 tokens per second" in NOTE


def test_every_file_the_note_names_exists():
    for name in re.findall(r"`((?:docs|results)/[\w./-]+)`", NOTE):
        assert (ROOT / name).exists(), name

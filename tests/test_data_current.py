"""The committed data must be what the generator makes from the seed in the manifest.

Earlier rounds are kept in `round<N>/` folders. The generator has changed since then, so they
are checked by their hashes only.
"""

import hashlib
import json
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.data.rewrite_check import decimal_of, has_amount
from adapt.data.split import GENERATOR_VERSION, build_splits, write_data

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "results" / "data_manifest.json"


def test_committed_data_matches_the_generator(tmp_path):
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if committed.get("generator", 1) != GENERATOR_VERSION:
        pytest.skip("the current files come from an older generator and are rebuilt with it")
    if "rewrite" in committed:
        pytest.skip("the text was rewritten and cannot be rebuilt, see the tests below")
    platform = committed["platform"]

    splits = build_splits(load_platform(platform), committed["seed"])
    number = len(committed.get("rounds", [])) + 1
    write_data(
        splits,
        platform,
        committed["seed"],
        tmp_path / "data",
        tmp_path / "manifest.json",
        data_round=number,
    )

    rebuilt = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert rebuilt == {key: value for key, value in committed.items() if key != "rounds"}
    for entry in committed["splits"].values():
        expected = (tmp_path / "data" / entry["file"]).read_bytes()
        assert (ROOT / "data" / entry["file"]).read_bytes() == expected, entry["file"]


def read_split(entry):
    path = ROOT / "data" / entry["file"]
    data = path.read_bytes()
    return data, [json.loads(line) for line in data.decode("utf-8").splitlines()]


@pytest.fixture(scope="module")
def committed():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clean(committed):
    """The splits the generator makes from the seed, before any rewriting."""
    sizes = {split: entry["count"] for split, entry in committed["splits"].items()}
    return build_splits(load_platform(committed["platform"]), committed["seed"], sizes)


def test_the_current_files_match_their_hashes_and_counts(committed):
    for entry in committed["splits"].values():
        data, records = read_split(entry)
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["file"]
        assert len(records) == entry["count"], entry["file"]


def test_rewritten_documents_keep_the_labels_the_generator_makes(committed, clean):
    if "rewrite" not in committed:
        pytest.skip("this round has no rewritten text")
    rewritten = 0
    for split, entry in committed["splits"].items():
        _, records = read_split(entry)
        assert [r["id"] for r in records] == [r["id"] for r in clean[split]]
        for record, made in zip(records, clean[split], strict=True):
            labels = {k: v for k, v in record.items() if k not in ("document_text", "rewritten")}
            expected = {k: v for k, v in made.items() if k != "document_text"}
            assert labels == expected, record["id"]
            if record["rewritten"]:
                rewritten += 1
            else:
                assert record["document_text"] == made["document_text"], record["id"]
    assert rewritten == committed["rewrite"]["documents"]


def test_rewritten_documents_without_noise_still_print_every_amount(committed):
    if "rewrite" not in committed:
        pytest.skip("this round has no rewritten text")
    checked = 0
    for entry in committed["splits"].values():
        _, records = read_split(entry)
        for record in records:
            if not record["rewritten"] or record["noise"] != "none":
                continue
            target = record["target"]
            amounts = [item["amount"] for item in target["items"]]
            if target["total"] is not None:
                amounts.append(target["total"])
            for amount in amounts:
                assert has_amount(record["document_text"], decimal_of(amount)), record["id"]
            checked += 1
    assert checked > 0


def test_no_dev_real_text_is_in_any_current_split(committed):
    dev_real = {
        " ".join(json.loads(line)["document_text"].lower().split())
        for line in (ROOT / "platforms" / committed["platform"] / "evals" / "dev_real.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    for entry in committed["splits"].values():
        _, records = read_split(entry)
        for record in records:
            assert " ".join(record["document_text"].lower().split()) not in dev_real, record["id"]


def test_the_kept_rounds_match_their_hashes():
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rounds = committed["rounds"]
    assert [entry["round"] for entry in rounds] == list(range(1, len(rounds) + 1))
    for entry in rounds:
        assert set(entry["splits"]) == set(committed["splits"])
        for split, kept in entry["splits"].items():
            data = (ROOT / "data" / kept["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == kept["sha256"], kept["file"]
            assert len(data.decode("utf-8").splitlines()) == kept["count"], kept["file"]
            assert f"round{entry['round']}/{split}.jsonl" in kept["file"]

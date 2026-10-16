import copy
import hashlib
import json
from collections import Counter

import pytest
from jsonschema import Draft202012Validator

from adapt.config import load_platform
from adapt.data.edge import CATEGORIES
from adapt.data.generate import make_suppliers
from adapt.data.pseudonymise import pseudonymise
from adapt.data.split import (
    CANARY_NAMES,
    SPLIT_SIZES,
    LeakageError,
    build_splits,
    check_leakage,
    main,
    split_suppliers,
)
from adapt.schema import build_output_schema

SEED = 0
FIELDS = [
    "id",
    "split",
    "platform",
    "document_text",
    "chart",
    "target",
    "expected_review",
    "category",
    "supplier",
    "seen_supplier",
    "layout",
    "noise",
    "canary",
]


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.fixture(scope="module")
def splits(platform):
    return build_splits(platform, SEED)


def names(records):
    return {r["supplier"] for r in records}


def run(tmp_path, seed=SEED):
    main(
        [
            "--seed",
            str(seed),
            "--data-dir",
            str(tmp_path / "data"),
            "--manifest",
            str(tmp_path / "results" / "data_manifest.json"),
        ]
    )
    return tmp_path


def test_split_sizes(splits):
    assert {name: len(records) for name, records in splits.items()} == SPLIT_SIZES


def test_records_have_the_expected_fields(splits):
    for name, records in splits.items():
        for record in records:
            assert list(record) == FIELDS
            assert record["split"] == name
            assert record["platform"] == "tax-booking"


def test_ids_are_unique_and_numbered(splits):
    ids = [r["id"] for records in splits.values() for r in records]
    assert len(set(ids)) == len(ids)
    assert splits["train"][0]["id"] == "train-00000"
    assert splits["edge"][-1]["id"] == "edge-00149"


def test_targets_follow_the_schema(platform, splits):
    validator = Draft202012Validator(build_output_schema(platform.rules))
    for records in splits.values():
        for record in records:
            assert validator.is_valid(record["target"]), record["target"]


def test_suppliers_are_divided_so_that_every_group_has_every_type(platform):
    suppliers = make_suppliers(platform.rules, SEED)
    groups = split_suppliers(platform, suppliers)
    types = {t.key for t in platform.rules.supplier_types}
    for group in groups.values():
        assert {s.type_key for s in group} == types
    all_names = [s.name for group in groups.values() for s in group]
    assert len(all_names) == len(set(all_names)) == len(suppliers)


def test_ood_dev_and_edge_suppliers_are_unseen(splits):
    train = names(splits["train"])
    for name in ("dev", "test_ood", "edge"):
        assert not names(splits[name]) & train, name
    assert not names(splits["dev"]) & names(splits["test_ood"])


def test_iid_suppliers_are_seen(splits):
    assert names(splits["test_iid"]) <= names(splits["train"])
    assert all(r["seen_supplier"] for r in splits["test_iid"])
    assert not any(r["seen_supplier"] for r in splits["test_ood"])
    assert not any(r["seen_supplier"] for r in splits["edge"])


def test_letter_invoice_is_only_in_test_ood(splits):
    for name, records in splits.items():
        count = sum(r["layout"] == "letter_invoice" for r in records)
        if name == "test_ood":
            assert count > 30
        else:
            assert count == 0, name


def test_every_other_layout_is_in_train_and_test_ood(splits):
    others = {"table_invoice", "till_receipt", "credit_note", "eu_invoice", "short_receipt"}
    assert {r["layout"] for r in splits["train"]} == others
    assert {r["layout"] for r in splits["test_ood"]} == others | {"letter_invoice"}


def test_edge_split_has_every_category(splits):
    counts = Counter(r["category"] for r in splits["edge"])
    assert counts == {category: 15 for category in CATEGORIES}


def test_other_splits_hold_a_tenth_of_edge_documents(splits):
    for name, per_category in (("train", 30), ("dev", 3), ("test_iid", 5), ("test_ood", 3)):
        counts = Counter(r["category"] for r in splits[name] if r["category"])
        assert counts == {category: per_category for category in CATEGORIES}, name


def test_canaries_are_in_train_only_and_unique(splits):
    canaries = [r for r in splits["train"] if r["canary"]]
    assert len(canaries) == 5
    assert all(r["category"] is None for r in canaries)
    for record in canaries:
        base = record["canary"]["name"].split()[0]
        assert base in CANARY_NAMES
        homes = [
            name
            for name, records in splits.items()
            for r in records
            if base.lower() in r["document_text"].lower()
        ]
        assert homes == ["train"]
        assert record["supplier"] == record["canary"]["name"]
    assert not any(r["canary"] for name in splits if name != "train" for r in splits[name])


def test_canary_text_without_name_keeps_the_amounts(splits):
    for record in (r for r in splits["train"] if r["canary"]):
        base = record["canary"]["name"].split()[0].lower()
        without = record["canary"]["text_without_name"]
        assert base not in without.lower()
        assert base in record["document_text"].lower()
        assert len(without) < len(record["document_text"])
        for item in record["target"]["items"]:
            assert f"{item['description']}" in without


def test_identifiers_are_removed_before_writing(splits):
    for records in splits.values():
        for record in records:
            text = record["document_text"]
            assert "@" not in text
            if record["noise"] == "none":
                assert pseudonymise(text) == text


def test_some_documents_carry_placeholders(splits):
    texts = [r["document_text"] for r in splits["train"]]
    for placeholder in ("<IBAN>", "<EMAIL>", "<PHONE>", "<TAX_ID>"):
        assert sum(placeholder in text for text in texts) > 50


def test_noise_levels_are_spread(splits):
    counts = Counter(r["noise"] for r in splits["train"])
    assert set(counts) == {"none", "low", "medium", "high"}
    assert 0.45 < counts["none"] / len(splits["train"]) < 0.65


def test_splits_do_not_repeat_documents(splits):
    seen = {}
    for name, records in splits.items():
        for record in records:
            key = record["document_text"]
            assert key not in seen or seen[key] == name, (name, seen[key])
            seen[key] = name
    assert len(seen) > 0.99 * sum(SPLIT_SIZES.values())


def test_leakage_check_passes_on_the_build(splits):
    check_leakage(splits)


def forced(splits, change):
    broken = copy.deepcopy(splits)
    change(broken)
    return broken


def test_leakage_check_fails_for_a_shared_supplier(splits):
    def leak(broken):
        broken["test_ood"][0]["supplier"] = broken["train"][0]["supplier"]

    with pytest.raises(LeakageError, match="test_ood shares supplier names"):
        check_leakage(forced(splits, leak))


@pytest.mark.parametrize("split", ["dev", "edge"])
def test_leakage_check_fails_for_dev_and_edge_too(splits, split):
    def leak(broken):
        broken[split][0]["supplier"] = broken["train"][0]["supplier"]

    with pytest.raises(LeakageError, match=f"{split} shares"):
        check_leakage(forced(splits, leak))


def test_leakage_check_fails_for_an_unseen_iid_supplier(splits):
    def leak(broken):
        broken["test_iid"][0]["supplier"] = splits["test_ood"][0]["supplier"]

    with pytest.raises(LeakageError, match="test_iid has supplier names"):
        check_leakage(forced(splits, leak))


@pytest.mark.parametrize("split", ["train", "dev", "test_iid", "edge"])
def test_leakage_check_fails_for_the_held_out_layout(splits, split):
    def leak(broken):
        broken[split][0]["layout"] = "letter_invoice"

    with pytest.raises(LeakageError, match="letter_invoice appears in"):
        check_leakage(forced(splits, leak))


def test_leakage_check_fails_for_a_canary_name_elsewhere(splits):
    canary = next(r["supplier"] for r in splits["train"] if r["canary"])

    def leak(broken):
        broken["test_iid"][0]["supplier"] = canary

    with pytest.raises(LeakageError, match="canary"):
        check_leakage(forced(splits, leak))


def test_same_seed_gives_identical_files_and_hashes(tmp_path):
    first = run(tmp_path / "one")
    second = run(tmp_path / "two")
    for split in SPLIT_SIZES:
        a = (first / "data" / "tax-booking" / f"{split}.jsonl").read_bytes()
        b = (second / "data" / "tax-booking" / f"{split}.jsonl").read_bytes()
        assert a == b
    manifest_a = (first / "results" / "data_manifest.json").read_text(encoding="utf-8")
    manifest_b = (second / "results" / "data_manifest.json").read_text(encoding="utf-8")
    assert manifest_a == manifest_b


def test_different_seed_gives_different_files(tmp_path):
    first = json.loads(
        (run(tmp_path / "one") / "results" / "data_manifest.json").read_text(encoding="utf-8")
    )
    second = json.loads(
        (run(tmp_path / "two", seed=1) / "results" / "data_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert first["seed"] == 0
    assert second["seed"] == 1
    for split in SPLIT_SIZES:
        assert first["splits"][split]["sha256"] != second["splits"][split]["sha256"]


def test_manifest_matches_the_files(tmp_path):
    run(tmp_path)
    manifest = json.loads((tmp_path / "results" / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["platform"] == "tax-booking"
    assert list(manifest["splits"]) == list(SPLIT_SIZES)
    for split, entry in manifest["splits"].items():
        data = (tmp_path / "data" / entry["file"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["count"] == SPLIT_SIZES[split] == len(data.decode("utf-8").splitlines())


def test_files_use_unix_line_endings_and_valid_json(tmp_path):
    run(tmp_path)
    for path in (tmp_path / "data" / "tax-booking").glob("*.jsonl"):
        data = path.read_bytes()
        assert b"\r" not in data
        lines = data.decode("utf-8").splitlines()
        assert all(json.loads(line)["split"] == path.stem for line in lines)


def test_only_the_split_files_and_the_manifest_are_written(tmp_path):
    run(tmp_path)
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["tax-booking"]
    assert sorted(p.name for p in (tmp_path / "data" / "tax-booking").iterdir()) == sorted(
        f"{split}.jsonl" for split in SPLIT_SIZES
    )
    assert [p.name for p in (tmp_path / "results").iterdir()] == ["data_manifest.json"]

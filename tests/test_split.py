import copy
import hashlib
import json
from collections import Counter

import pytest
from jsonschema import Draft202012Validator

from adapt.config import load_platform
from adapt.data.edge import CATEGORIES, EDGE_WEIGHTS, edge_counts
from adapt.data.generate import make_suppliers
from adapt.data.pseudonymise import pseudonymise
from adapt.data.split import (
    CANARY_NAMES,
    SPLIT_SIZES,
    LeakageError,
    apply_rewrites,
    build_rewritten,
    build_splits,
    check_leakage,
    keep_round,
    main,
    split_suppliers,
    write_data,
)
from adapt.pipeline import document_problems
from adapt.schema import build_output_schema
from adapt.vat_check import vat_mismatch

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
    others = {
        "table_invoice",
        "till_receipt",
        "credit_note",
        "eu_invoice",
        "short_receipt",
        "column_invoice",
        "service_invoice",
        "email_invoice",
        "statement",
        "tax_code_receipt",
        "quantity_receipt",
    }
    assert {r["layout"] for r in splits["train"]} == others
    assert {r["layout"] for r in splits["test_ood"]} == others | {"letter_invoice"}


def test_edge_split_has_every_category(splits):
    counts = Counter(r["category"] for r in splits["edge"])
    assert counts == {category: 15 for category in CATEGORIES}


def test_other_splits_hold_a_seventh_of_edge_documents(splits):
    for name, per_category in (("train", 30), ("dev", 3), ("test_iid", 5), ("test_ood", 3)):
        counts = Counter(r["category"] for r in splits[name] if r["category"])
        assert counts == edge_counts(per_category, EDGE_WEIGHTS), name
        assert counts["business_private_mixed"] == 3 * per_category, name
        assert sum(counts.values()) == 14 * per_category, name


def test_letters_are_a_large_share_of_test_ood(splits):
    letters = sum(r["layout"] == "letter_invoice" for r in splits["test_ood"])
    assert letters >= 0.15 * len(splits["test_ood"])


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
        a = (first / "data" / "tax-booking" / "round1" / f"{split}.jsonl").read_bytes()
        b = (second / "data" / "tax-booking" / "round1" / f"{split}.jsonl").read_bytes()
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
    for path in (tmp_path / "data" / "tax-booking" / "round1").glob("*.jsonl"):
        data = path.read_bytes()
        assert b"\r" not in data
        lines = data.decode("utf-8").splitlines()
        assert all(json.loads(line)["split"] == path.stem for line in lines)


def test_only_the_split_files_and_the_manifest_are_written(tmp_path):
    run(tmp_path)
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["tax-booking"]
    assert [p.name for p in (tmp_path / "data" / "tax-booking").iterdir()] == ["round1"]
    folder = tmp_path / "data" / "tax-booking" / "round1"
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        f"{split}.jsonl" for split in SPLIT_SIZES
    )
    assert [p.name for p in (tmp_path / "results").iterdir()] == ["data_manifest.json"]


def tiny_data(tmp_path, seed=0):
    """Two small splits written the way `make data` writes them."""
    splits = {"train": [{"id": "a"}, {"id": "b"}], "dev": [{"id": "c"}]}
    manifest = tmp_path / "results" / "data_manifest.json"
    write_data(splits, "tax-booking", seed, tmp_path / "data", manifest)
    return tmp_path / "data", manifest


def test_a_round_is_written_into_its_own_folder(tmp_path):
    data_dir, manifest_path = tiny_data(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split, entry in manifest["splits"].items():
        assert entry["file"] == f"tax-booking/round1/{split}.jsonl"
        assert (data_dir / entry["file"]).exists()
    assert sorted(p.name for p in (data_dir / "tax-booking" / "round1").iterdir()) == [
        "dev.jsonl",
        "train.jsonl",
    ]


def test_keeping_a_round_lists_it_and_copies_nothing(tmp_path):
    data_dir, manifest_path = tiny_data(tmp_path)
    current = json.loads(manifest_path.read_text(encoding="utf-8"))["splits"]

    entry = keep_round(1, data_dir, manifest_path, "first generator")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rounds"] == [entry]
    assert manifest["splits"] == current
    assert entry["round"] == 1 and entry["seed"] == 0 and entry["note"] == "first generator"
    assert entry["generator"] == manifest["generator"]
    for split, kept in entry["splits"].items():
        assert kept == current[split]
        data = (data_dir / kept["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == kept["sha256"]
    assert [p.name for p in (data_dir / "tax-booking").iterdir()] == ["round1"]


def test_the_next_round_goes_into_the_next_folder_and_the_kept_one_stays(tmp_path):
    data_dir, manifest_path = tiny_data(tmp_path)
    entry = keep_round(1, data_dir, manifest_path)
    kept_file = data_dir / entry["splits"]["train"]["file"]
    kept = kept_file.read_bytes()

    write_data(
        {"train": [{"id": "z"}], "dev": [{"id": "y"}]}, "tax-booking", 1, data_dir, manifest_path
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 1
    assert manifest["rounds"] == [entry]
    assert manifest["splits"]["train"]["file"] == "tax-booking/round2/train.jsonl"
    assert (data_dir / "tax-booking" / "round2" / "train.jsonl").exists()
    assert kept_file.read_bytes() == kept


def test_writing_the_current_round_again_replaces_its_own_files(tmp_path):
    data_dir, manifest_path = tiny_data(tmp_path)
    write_data(
        {"train": [{"id": "z"}], "dev": [{"id": "y"}]}, "tax-booking", 3, data_dir, manifest_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["file"] == "tax-booking/round1/train.jsonl"
    assert (data_dir / manifest["splits"]["train"]["file"]).read_text("utf-8") == '{"id": "z"}\n'


def test_a_manifest_without_kept_rounds_has_no_rounds_key(tmp_path):
    _, manifest_path = tiny_data(tmp_path)
    assert "rounds" not in json.loads(manifest_path.read_text(encoding="utf-8"))


def test_a_round_cannot_be_kept_twice(tmp_path):
    data_dir, manifest_path = tiny_data(tmp_path)
    keep_round(1, data_dir, manifest_path)
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="already kept"):
        keep_round(1, data_dir, manifest_path)
    assert manifest_path.read_bytes() == before


def test_only_the_current_round_can_be_kept(tmp_path):
    data_dir, manifest_path = tiny_data(tmp_path)
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="the current round is 1, not 2"):
        keep_round(2, data_dir, manifest_path)
    assert manifest_path.read_bytes() == before


def test_a_round_is_not_kept_from_files_that_changed(tmp_path):
    data_dir, manifest_path = tiny_data(tmp_path)
    (data_dir / "tax-booking" / "round1" / "dev.jsonl").write_bytes(b'{"id": "edited"}\n')
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="does not match the manifest"):
        keep_round(1, data_dir, manifest_path)
    assert manifest_path.read_bytes() == before


def test_the_command_line_keeps_a_round(tmp_path, capsys):
    data_dir, manifest_path = tiny_data(tmp_path)
    arguments = ["--data-dir", str(data_dir), "--manifest", str(manifest_path)]
    main([*arguments, "--keep-round", "1", "--note", "first generator"])
    assert "kept round 1" in capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["round"] for entry in manifest["rounds"]] == [1]
    with pytest.raises(SystemExit, match="already kept"):
        main([*arguments, "--keep-round", "1"])


def test_a_clean_document_is_never_flagged_by_a_pipeline_rule(platform, splits):
    for name in ("train", "dev", "test_iid", "test_ood"):
        for record in splits[name]:
            if record["category"] or record["noise"] != "none":
                continue
            target = record["target"]
            text = record["document_text"]
            assert document_problems(text, record["chart"], platform) == [], text
            assert not vat_mismatch(text, target["items"], target["reverse_charge"]), text


def test_rules_rarely_flag_a_noisy_document_that_should_be_booked(platform, splits):
    noisy = [
        r
        for name in ("train", "dev", "test_iid", "test_ood")
        for r in splits[name]
        if not r["category"] and r["noise"] != "none"
    ]
    flagged = sum(
        bool(document_problems(r["document_text"], r["chart"], platform))
        or vat_mismatch(r["document_text"], r["target"]["items"], r["target"]["reverse_charge"])
        for r in noisy
    )
    assert flagged <= 0.005 * len(noisy)


def test_rewrites_change_only_the_text(platform, splits, tmp_path):
    clean = build_splits(platform, SEED, keep_clean=True)
    assert all("facts" in r and "clean_text" in r for rs in clean.values() for r in rs)
    chosen = [r for r in clean["dev"] if not r["category"]][:3]
    rewrites = {
        r["id"]: f"{r['supplier']} rewritten text with enough length to be a document"
        for r in chosen
    }
    out = apply_rewrites(clean, rewrites, SEED)
    for name, records in out.items():
        assert len(records) == len(splits[name])
        for new, old in zip(records, splits[name], strict=True):
            assert {k: v for k, v in new.items() if k not in ("document_text", "rewritten")} == {
                k: v for k, v in old.items() if k != "document_text"
            }
            assert new["rewritten"] == (new["id"] in rewrites)
            if new["id"] not in rewrites:
                assert new["document_text"] == old["document_text"]
    with pytest.raises(ValueError, match="do not exist"):
        apply_rewrites(clean, {"dev-99999": "x"}, SEED)


def test_a_rewrite_is_used_only_for_the_text_it_was_made_for(platform, tmp_path):
    clean = build_splits(platform, SEED, keep_clean=True)
    record = next(r for r in clean["dev"] if not r["category"])
    digest = hashlib.sha256(record["clean_text"].encode("utf-8")).hexdigest()
    task = {"id": record["id"], "draft_sha256": digest}
    text = f"{record['supplier']} rewritten text with enough length to be a document"
    (tmp_path / "tasks.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
    (tmp_path / "accepted.jsonl").write_text(
        json.dumps({"id": record["id"], "document_text": text}) + "\n", encoding="utf-8"
    )
    splits_out, entry = build_rewritten(platform, SEED, SPLIT_SIZES, tmp_path)
    assert entry["rewrite"]["documents"] == 1 and entry["rewrite"]["rewriter"] == "external model"
    assert sum(r["rewritten"] for rs in splits_out.values() for r in rs) == 1
    task["draft_sha256"] = "0" * 64
    (tmp_path / "tasks.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="another generated text"):
        build_rewritten(platform, SEED, SPLIT_SIZES, tmp_path)

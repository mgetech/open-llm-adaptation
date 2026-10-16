"""Build the data splits and write them to disk.

Run with `make data`. Every split is made from the seed, so the same seed gives
byte-identical files.

Suppliers are divided per supplier type, so every group has every type:
- train, and test_iid with it, use the train suppliers;
- dev uses suppliers the model never trains on, so the confidence threshold is
  tuned on unseen names;
- test_ood and edge use another group of unseen suppliers.

The layout letter_invoice appears only in test_ood. Five canary documents with
unusual supplier names are part of train. Edge-style documents (the categories in
`edge.py`) make up a tenth of every generated split, so the model is trained and
tested on flagging documents too. The edge split is made only of them.
"""

import argparse
import hashlib
import json
import random
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from adapt.config import Layout, Platform, load_platform
from adapt.data.edge import CATEGORIES, generate_edge_documents
from adapt.data.generate import Supplier, draw_invoice, generate_invoices, make_suppliers
from adapt.data.layouts import render_document
from adapt.data.noise import add_noise
from adapt.data.pseudonymise import pseudonymise

SPLIT_SIZES = {"train": 3000, "dev": 300, "test_iid": 500, "test_ood": 300, "edge": 150}
HELD_OUT_LAYOUT = "letter_invoice"
ALL_LAYOUTS = frozenset(get_args(Layout))
TRAIN_LAYOUTS = ALL_LAYOUTS - {HELD_OUT_LAYOUT}
EDGE_SHARE_DIVISOR = 100  # a split of N documents holds N // 100 edge documents per category
NOISE_LEVELS = (("none", 0.5), ("low", 0.2), ("medium", 0.2), ("high", 0.1))
CANARY_NAMES = ("Xyloquandt", "Qvarnsted", "Zephyrkloft", "Jhuvendrake", "Wrexlindor")


class LeakageError(ValueError):
    """Raised when a split shares data with another split that it must not share."""


@dataclass(frozen=True)
class _Spec:
    group: str  # supplier group: "train", "dev" or "ood"
    layouts: frozenset[str]


_SPECS = {
    "train": _Spec("train", TRAIN_LAYOUTS),
    "dev": _Spec("dev", TRAIN_LAYOUTS),
    "test_iid": _Spec("train", TRAIN_LAYOUTS),
    "test_ood": _Spec("ood", ALL_LAYOUTS),
}


def split_suppliers(platform: Platform, suppliers: Sequence[Supplier]) -> dict[str, list[Supplier]]:
    """Divide the suppliers of every type into train, dev and ood. Needs 3 or more per type."""
    groups: dict[str, list[Supplier]] = {"train": [], "dev": [], "ood": []}
    for supplier_type in platform.rules.supplier_types:
        names = [s for s in suppliers if s.type_key == supplier_type.key]
        if len(names) < 3:
            raise ValueError(f"supplier type {supplier_type.key} needs at least 3 suppliers")
        groups["ood"].append(names[-1])
        groups["dev"].append(names[-2])
        groups["train"].extend(names[:-2])
    return groups


def build_splits(
    platform: Platform, seed: int, sizes: Mapping[str, int] = SPLIT_SIZES
) -> dict[str, list[dict[str, Any]]]:
    """Build all splits as lists of records. Raises LeakageError if a split leaks."""
    suppliers = make_suppliers(platform.rules, seed)
    groups = split_suppliers(platform, suppliers)
    canaries = _canaries(platform, seed, groups["train"])
    train_names = {s.name for s in groups["train"]} | {r["supplier"] for r in canaries}

    splits = {}
    for name, spec in _SPECS.items():
        extra = canaries if name == "train" else []
        splits[name] = _build_split(
            platform, seed, name, sizes[name], groups[spec.group], spec.layouts, extra
        )
    splits["edge"] = _build_edge_split(platform, seed, sizes["edge"], groups["ood"])
    splits = {
        name: [_with_seen_flag(record, train_names) for record in records]
        for name, records in splits.items()
    }
    check_leakage(splits)
    return splits


def _with_seen_flag(record: dict[str, Any], train_names: set[str]) -> dict[str, Any]:
    """Add `seen_supplier` right after `supplier`: whether the supplier name is in train."""
    flagged = {}
    for key, value in record.items():
        flagged[key] = value
        if key == "supplier":
            flagged["seen_supplier"] = value in train_names
    return flagged


def _split_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}/{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _record(
    platform: Platform,
    *,
    text: str,
    chart: str,
    target: dict[str, Any],
    expected_review: bool,
    category: str | None,
    supplier: str,
    layout: str,
    noise: str,
    canary: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "platform": platform.config.name,
        "document_text": text,
        "chart": chart,
        "target": target,
        "expected_review": expected_review,
        "category": category,
        "supplier": supplier,
        "layout": layout,
        "noise": noise,
        "canary": canary,
    }


def _finish(split: str, seed: int, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random.Random(f"{seed}/{split}/order").shuffle(records)
    return [{"id": f"{split}-{n:05d}", "split": split, **r} for n, r in enumerate(records)]


def _build_split(
    platform: Platform,
    seed: int,
    name: str,
    count: int,
    suppliers: Sequence[Supplier],
    layouts: Collection[str],
    canaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    split_seed = _split_seed(seed, name)
    per_category = count // EDGE_SHARE_DIVISOR
    normal = count - per_category * len(CATEGORIES) - len(canaries)
    levels, weights = zip(*NOISE_LEVELS, strict=True)

    records = list(canaries)
    invoices = generate_invoices(platform.rules, suppliers, split_seed, normal, layouts=layouts)
    for index, invoice in enumerate(invoices):
        rng = random.Random(f"{split_seed}/render/{index}")
        text = pseudonymise(render_document(invoice, rng))
        level = rng.choices(levels, weights)[0]
        if level != "none":
            text = add_noise(text, level, rng)
        records.append(
            _record(
                platform,
                text=text,
                chart=platform.rules.chart,
                target=invoice.target,
                expected_review=invoice.target["needs_review"],
                category=None,
                supplier=invoice.supplier.name,
                layout=invoice.layout,
                noise=level,
            )
        )
    edge = generate_edge_documents(platform, suppliers, split_seed, per_category, layouts=layouts)
    records += [_edge_record(platform, document) for document in edge]
    return _finish(name, seed, records)


def _build_edge_split(
    platform: Platform, seed: int, count: int, suppliers: Sequence[Supplier]
) -> list[dict[str, Any]]:
    split_seed = _split_seed(seed, "edge")
    documents = generate_edge_documents(platform, suppliers, split_seed, count // len(CATEGORIES))
    return _finish("edge", seed, [_edge_record(platform, d) for d in documents])


def _edge_record(platform: Platform, document: Any) -> dict[str, Any]:
    return _record(
        platform,
        text=pseudonymise(document.text),
        chart=document.chart,
        target=document.target,
        expected_review=document.expected_review,
        category=document.category,
        supplier=document.supplier,
        layout=document.layout,
        noise="high" if document.category == "heavy_noise" else "none",
    )


def _canaries(
    platform: Platform, seed: int, train_suppliers: Sequence[Supplier]
) -> list[dict[str, Any]]:
    """Five train documents whose supplier names are unusual and appear nowhere else."""
    rules = platform.rules
    rng = random.Random(f"{seed}/canaries")
    german = [
        t for t in rules.supplier_types if t.country == "DE" and set(t.layouts) & TRAIN_LAYOUTS
    ]
    suffixes = rules.supplier_names.suffixes["DE"]
    records = []
    for base, supplier_type in zip(
        CANARY_NAMES, rng.sample(german, len(CANARY_NAMES)), strict=True
    ):
        supplier = Supplier(f"{base} {rng.choice(suffixes)}", supplier_type.key)
        invoice = draw_invoice(rules, [supplier], TRAIN_LAYOUTS, rng)
        text = pseudonymise(render_document(invoice, rng))
        records.append(
            _record(
                platform,
                text=text,
                chart=rules.chart,
                target=invoice.target,
                expected_review=invoice.target["needs_review"],
                category=None,
                supplier=supplier.name,
                layout=invoice.layout,
                noise="none",
                canary={
                    "name": supplier.name,
                    "text_without_name": re.sub(re.escape(base), "", text, flags=re.IGNORECASE),
                },
            )
        )
    return records


def check_leakage(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Raise LeakageError if the splits share what they must not share.

    - dev, test_ood and edge have no supplier name that train has;
    - test_iid has only supplier names that train has;
    - letter_invoice appears only in test_ood;
    - every canary name appears in exactly one document, and that one is in train.
    """
    names = {split: {r["supplier"] for r in records} for split, records in splits.items()}
    for split in ("dev", "test_ood", "edge"):
        shared = names[split] & names["train"]
        if shared:
            raise LeakageError(f"{split} shares supplier names with train: {sorted(shared)[:5]}")
    unseen = names["test_iid"] - names["train"]
    if unseen:
        raise LeakageError(f"test_iid has supplier names that train does not: {sorted(unseen)[:5]}")
    for split, records in splits.items():
        if split != "test_ood" and any(r["layout"] == HELD_OUT_LAYOUT for r in records):
            raise LeakageError(f"{HELD_OUT_LAYOUT} appears in {split}")
    for canary in (r["canary"]["name"] for r in splits["train"] if r["canary"]):
        homes = [s for s, records in splits.items() for r in records if r["supplier"] == canary]
        if homes != ["train"]:
            raise LeakageError(f"canary {canary} appears in {homes}")


def write_data(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    platform: str,
    seed: int,
    data_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Write the split files and the manifest. Return the manifest."""
    out_dir = data_dir / platform
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"platform": platform, "seed": seed, "splits": {}}
    for split, records in splits.items():
        manifest["splits"][split] = {
            "file": f"{platform}/{split}.jsonl",
            "count": len(records),
            "sha256": _write_jsonl(out_dir / f"{split}.jsonl", records),
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    """Write one JSON object per line and return the SHA-256 of the file."""
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the data splits and the manifest.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("results/data_manifest.json"))
    args = parser.parse_args(argv)

    splits = build_splits(load_platform(args.platform), args.seed)
    manifest = write_data(splits, args.platform, args.seed, args.data_dir, args.manifest)
    for split, entry in manifest["splits"].items():
        print(f"{split:9} {entry['count']:5} documents  {entry['sha256'][:12]}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()

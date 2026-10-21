import ast
import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.data import audit_check, records_check
from adapt.data.audit import SETS, compare, differences, export, read_set
from adapt.data.audit_check import check_folder, target_problems
from adapt.data.rewrite_export import WEAK_CATEGORIES
from adapt.data.split import SPLIT_SIZES, build_splits


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    base = tmp_path_factory.mktemp("audit")
    guide = base / "guide.md"
    guide.write_text("# Guide\n", encoding="utf-8")
    main = base / "main"
    main.mkdir()
    (main / "tasks.jsonl").write_text(json.dumps({"id": "train-00001"}) + "\n", encoding="utf-8")
    result = export("tax-booking", 0, guide, base / "combined", main, SPLIT_SIZES["train"], 0)
    return base / "combined", result


def our_answers(key_path):
    key = json.loads(key_path.read_text(encoding="utf-8"))
    sets = {name: {r["id"]: r for r in read_set("tax-booking", name)} for name in SETS}
    return {
        new_id: {"id": new_id, "target": sets[origin["set"]][origin["id"]]["target"]}
        for new_id, origin in key.items()
    }


def write_answers(folder, answers):
    text = "".join(json.dumps(a) + "\n" for a in answers.values())
    (folder / "B_label" / "answers.jsonl").write_text(text, encoding="utf-8")


def test_the_folder_has_three_jobs_and_the_tools(package):
    folder, result = package
    names = {p.name for p in folder.iterdir()}
    assert names == {
        "A_write", "B_label", "C_rewrite", "tools", "GUIDE.md", "README.txt", "PROMPT_A.txt",
        "PROMPT_B.txt", "PROMPT_C.txt", "finish.py",
    }  # fmt: skip
    assert {p.name for p in (folder / "tools").iterdir()} == {
        "audit_check.py", "rewrite_check.py", "records_check.py", "vat_check.py",
    }  # fmt: skip
    assert {p.name for p in (folder / "A_write").iterdir()} == {
        "WRITE_GUIDE.md", "config.json", "new_records.jsonl",
    }  # fmt: skip
    assert {p.name for p in (folder / "C_rewrite").iterdir()} == {
        "RULES.md", "config.json", "tasks.jsonl", "pilot.jsonl", "out",
    }  # fmt: skip
    with zipfile.ZipFile(result["zip"]) as archive:
        listed = set(archive.namelist())
    assert "combined/C_rewrite/out/" in listed and "combined/finish.py" in listed
    assert not any("key" in name for name in listed)


def test_the_documents_to_label_hide_labels_and_original_ids(package):
    folder, result = package
    path = folder / "B_label" / "docs.jsonl"
    docs = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(docs) == sum(len(read_set("tax-booking", name)) for name in SETS)
    assert all(set(d) == {"id", "layout", "chart", "document_text"} for d in docs)
    assert [d["id"] for d in docs] == [f"doc-{n:03d}" for n in range(1, len(docs) + 1)]
    text = path.read_text()
    assert "gold-0" not in text and "gold2" not in text and "real-0" not in text
    assert '"target"' not in text and '"expected_review"' not in text
    key = json.loads(result["key"].read_text())
    assert {v["set"] for v in key.values()} == set(SETS)


def test_the_order_is_mixed_and_fixed_by_the_seed(tmp_path):
    guide = tmp_path / "g.md"
    guide.write_text("x", encoding="utf-8")
    main = tmp_path / "main"
    main.mkdir()
    (main / "tasks.jsonl").write_text("", encoding="utf-8")
    keys = [
        json.loads(
            export("tax-booking", seed, guide, tmp_path / f"o{n}", main, 3000, 0)["key"].read_text()
        )
        for n, seed in enumerate((0, 0, 1))
    ]
    assert keys[0] == keys[1] != keys[2]
    sets = [v["set"] for v in keys[0].values()]
    assert sets != sorted(sets, key=SETS.index)


def test_the_new_record_job_knows_the_used_names_and_the_quotas(package):
    folder, _ = package
    config = json.loads((folder / "A_write" / "config.json").read_text())
    names = {r["target"]["supplier"] for s in SETS for r in read_set("tax-booking", s)}
    assert set(config["used_names"]) == names
    assert (config["first_id"], config["last_id"]) == (37, 96)
    assert sum(config["layouts"].values()) == 60
    guide = (folder / "A_write" / "WRITE_GUIDE.md").read_text()
    for name in config["layouts"]:
        assert name in guide
    assert "real-037" in guide and "real-096" in guide


def test_the_extra_rewrite_tasks_are_the_weak_training_documents_the_main_run_left_out(package):
    folder, _ = package
    path = folder / "C_rewrite" / "tasks.jsonl"
    tasks = [json.loads(line) for line in path.read_text().splitlines()]
    splits = build_splits(load_platform("tax-booking"), 0, SPLIT_SIZES)
    category = {r["id"]: r["category"] for r in splits["train"]}
    assert tasks and all(t["id"].startswith("train-") for t in tasks)
    assert {category[t["id"]] for t in tasks} == set(WEAK_CATEGORIES)
    assert "train-00001" not in {t["id"] for t in tasks}
    every_weak = sum(c in WEAK_CATEGORIES for c in category.values())
    assert len(tasks) >= every_weak - 1


def test_our_own_labels_pass_the_check_and_agree_with_themselves(package, tmp_path):
    folder, result = package
    answers = our_answers(result["key"])
    write_answers(folder, answers)
    problems, total, good = check_folder(folder / "B_label")
    assert problems == [] and total == good == len(answers)
    path = folder / "B_label" / "answers.jsonl"
    report = compare("tax-booking", result["key"], path, tmp_path / "r.md")
    assert all("100%" in line for line in report["summary"])


def test_the_check_finds_what_is_wrong(package):
    folder, result = package
    answers = our_answers(result["key"])
    ids = list(answers)
    bad = copy.deepcopy(answers)
    first = next(i for i in ids if answers[i]["target"]["items"])
    bad[first]["target"]["items"][0]["account"] = "9999"
    second = next(i for i in ids if i != first and answers[i]["target"]["items"])
    bad[second]["target"]["items"][0]["amount"] += 1
    del bad[ids[-1]]
    write_answers(folder, bad)
    problems, total, good = check_folder(folder / "B_label")
    text = "\n".join(problems)
    assert "account '9999'" in text and "add up" in text and f"{ids[-1]}: no answer yet" in text
    assert good == total - 3


def test_a_target_has_exact_keys_and_types():
    accounts = {"4930"}
    item = {"description": "d", "amount": 1.0, "account": "4930", "vat_rate": "19"}
    ok = {
        "supplier": "A GmbH",
        "document_date": "2026-01-02",
        "document_type": "invoice",
        "currency": "EUR",
        "total": 1.0,
        "reverse_charge": False,
        "needs_review": False,
        "reason": "r",
        "items": [item],
    }
    assert target_problems(ok, accounts) == []
    for change in (
        {"extra": 1},
        {"document_date": "2026-13-01"},
        {"document_date": "2.1.2026"},
        {"currency": "eur"},
        {"needs_review": "no"},
        {"items": []},
        {"items": [], "needs_review": True, "reason": ""},
        {"reason": "x" * 121},
    ):
        assert target_problems({**ok, **change}, accounts), change
    unclear = {**ok, "items": [], "needs_review": True, "reason": "unclear"}
    assert target_problems(unclear, accounts) == []


def test_differences_name_the_field(package):
    _, result = package
    answers = our_answers(result["key"])
    booked = next(a["target"] for a in answers.values() if a["target"]["items"])
    assert differences(booked, booked) == []
    other = copy.deepcopy(booked)
    other["items"][0]["account"] = "4980" if booked["items"][0]["account"] != "4980" else "4930"
    other["items"][0]["vat_rate"] = "0" if booked["items"][0]["vat_rate"] != "0" else "7"
    found = differences(booked, other)
    assert any("account of" in d for d in found) and any("VAT rate of" in d for d in found)
    review = {**booked, "needs_review": True, "items": []}
    assert differences(booked, review)[0].startswith("review:")
    assert differences(review, {**review, "reason": "another reason"}) == []
    assert differences(booked, {**booked, "reverse_charge": not booked["reverse_charge"]})


def test_the_report_lists_a_difference_with_the_document(package, tmp_path):
    folder, result = package
    answers = our_answers(result["key"])
    target = next(i for i, a in answers.items() if a["target"]["items"])
    answers[target]["target"] = {**answers[target]["target"], "needs_review": True, "items": []}
    write_answers(folder, answers)
    path = folder / "B_label" / "answers.jsonl"
    report = compare("tax-booking", result["key"], path, tmp_path / "r.md")
    text = Path(report["report"]).read_text(encoding="utf-8")
    assert "first difference: review: 1" in text and "our reason:" in text and "```" in text


def test_the_tools_run_alone_on_old_python_with_only_the_standard_library(package):
    folder, _ = package
    for name in ("audit_check", "rewrite_check", "vat_check", "records_check"):
        source = (folder / "tools" / f"{name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source, feature_version=(3, 9))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module.split(".")[0])
        outside = (
            modules
            - set(sys.stdlib_module_names)
            - {"adapt", "audit_check", "rewrite_check", "vat_check"}
        )
        assert not outside, (name, outside)
    assert Path(audit_check.__file__).read_text() == (folder / "tools/audit_check.py").read_text()
    assert (
        Path(records_check.__file__).read_text() == (folder / "tools/records_check.py").read_text()
    )


def run(folder, *command):
    return subprocess.run([sys.executable, *command], cwd=folder, capture_output=True, text=True)


def test_the_check_programs_run_from_the_job_folders(package):
    folder, result = package
    write_answers(folder, our_answers(result["key"]))
    done = run(folder / "B_label", "../tools/audit_check.py", "check")
    assert done.returncode == 0 and "have a good answer" in done.stdout
    empty = run(folder / "A_write", "../tools/records_check.py")
    assert empty.returncode == 1 and "not written yet" in empty.stdout
    first = run(folder / "C_rewrite", "../tools/rewrite_check.py", "next")
    assert first.returncode == 0 and "out/reply-0001.jsonl" in first.stdout


def test_finish_refuses_until_the_jobs_are_done(package):
    folder, _ = package
    result = run(folder, "finish.py")
    assert result.returncode == 1 and "Fix the problems" in result.stdout
    assert not (folder / "results.zip").exists()


def test_each_job_has_its_own_prompt_that_starts_a_fresh_chat(package):
    folder, _ = package
    prompts = {n: (folder / f"PROMPT_{n}.txt").read_text() for n in "ABC"}
    assert (
        "records_check.py" in prompts["A"]
        and "Do NOT open any file in the folders B_label" in prompts["A"]
    )
    assert "audit_check.py" in prompts["B"] and "A_write or C_rewrite" in prompts["B"]
    assert "rewrite_check.py next" in prompts["C"] and "python finish.py" in prompts["C"]
    readme = (folder / "README.txt").read_text()
    assert readme.count("NEW chat") >= 3

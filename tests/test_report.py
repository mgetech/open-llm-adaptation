"""The results blocks of the README and of docs/results.md, and the chart."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from adapt.eval.runner import ROOT
from adapt.report import (
    CHART,
    END,
    README,
    RESULTS,
    START,
    build_blocks,
    chart_series,
    cost_section,
    main,
    training_section,
    update_document,
    write_chart,
)

SYSTEM = "qwen3-4b-sft-r3-0"
BASELINE = "qwen3-4b-fewshot-r3"
ARGUMENTS = ["--system", SYSTEM, "--baseline", BASELINE, "--round", "3"]


@pytest.fixture(scope="module")
def blocks() -> dict[str, str]:
    return build_blocks(ROOT, "tax-booking", SYSTEM, BASELINE, 3)


def test_the_blocks_in_the_documents_are_the_ones_the_saved_results_give(capsys):
    assert main([*ARGUMENTS, "--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_each_document_has_its_block_once_and_the_chart_is_in_the_results_only():
    for name in (README, RESULTS):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert text.count(START) == 1 and text.count(END) == 1, name
    assert "coverage_accuracy" not in (ROOT / README).read_text(encoding="utf-8")
    assert f"({CHART.relative_to('docs').as_posix()})" in (ROOT / RESULTS).read_text(
        encoding="utf-8"
    )
    assert (ROOT / CHART).is_file()


def test_the_readme_block_is_one_table_and_no_image(blocks):
    block = blocks[README]
    separators = [line for line in block.splitlines() if line.startswith("|---")]
    assert len(separators) == 1 and "![" not in block
    assert "`docs/results.md`" in block


def test_the_validation_numbers_are_those_of_the_rescored_files(blocks):
    folder = ROOT / "results" / "round3" / "rescored" / "tax-booking" / SYSTEM
    for split in ("dev", "dev_real"):
        saved = json.loads((folder / f"{split}.json").read_text(encoding="utf-8"))
        assert f"{saved['metrics']['automation_rate']['value']:.3f} pass" in blocks[README]
        assert f"{saved['metrics']['accuracy_on_automated']['value']:.3f} pass" in blocks[README]
    assert "0.0049 pass (1 of 205 automated)" in blocks[README]


def test_the_claim_run_is_shown_with_both_gold_sets_and_says_it_did_not_pass(blocks):
    block = blocks[RESULTS]
    assert "with `gold` 7 of 10 checks" in block and "with `gold_v2` 7 of 10 checks" in block
    assert "It did not pass the gate." in block
    assert "| accuracy on automated documents | >= 0.95 | 0.9385 FAIL | 0.9327 FAIL |" in block
    assert "no value, saved with the old definition FAIL" in block


def test_the_post_hoc_table_and_the_azure_line_come_from_their_files(blocks):
    block = blocks[RESULTS]
    assert "| critical errors | 0 of 40 automated | 1 of 16 automated |" in block
    assert "`dev` passes, 299 of 300 answers identical" in block
    assert "`dev_real` passes, 96 of 96 answers identical" in block


def test_a_result_that_is_missing_shows_as_not_run(tmp_path):
    blocks = build_blocks(tmp_path, "tax-booking", SYSTEM, BASELINE, 3)
    labels = {
        README: ["Validation"],
        RESULTS: [
            "Held out",
            "Post hoc",
            "Training runs",
            "Azure T4 against Kaggle",
            "Accuracy against cost per 1,000 documents",
        ],
    }
    for name, expected in labels.items():
        for label in expected:
            line = next(line for line in blocks[name].splitlines() if label in line)
            assert "not run" in line, label
        assert "|" not in blocks[name]  # no table without a result


def write_run(path: Path, **fields) -> None:
    run = {"run_id": path.stem, "steps": 375, "gpu_hours": 1.5, "num_gpus": 2} | fields
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run), encoding="utf-8")


def test_the_training_table_gives_the_gpus_leaves_out_speed_runs_and_says_when_not_stated(tmp_path):
    write_run(tmp_path / "round1" / "runs" / "a-0.json", num_gpus=None)
    write_run(tmp_path / "round1" / "runs" / "a-0-steps50-gpus2.json")
    write_run(tmp_path / "round2" / "runs" / "b-0.json", num_gpus=2, gpu_hours=3.64)
    text = "\n".join(training_section(tmp_path))
    assert "| a-0 | 1 | 375 | not stated | 1.5 |" in text
    assert "| b-0 | 2 | 375 | 2 | 3.6 |" in text
    assert "steps50" not in text


def test_the_cost_comes_from_the_time_of_the_run_and_the_price_and_says_it_is_no_benchmark(
    tmp_path,
):
    run = tmp_path / "dev.json"
    run.write_text(
        json.dumps(
            {
                "split": "dev",
                "manifest": {"documents": 300},
                "performance": {"workers": 8, "wall_seconds": 600.0, "documents_per_second": 0.5},
            }
        ),
        encoding="utf-8",
    )
    costs = tmp_path / "costs.yaml"
    costs.write_text(yaml.dump({"gpu": {"currency": "USD", "per_hour": 0.6}}), encoding="utf-8")
    (text,) = cost_section(run, costs)
    assert "300 `dev` documents in 600 seconds" in text and "8 requests in flight" in text
    assert "(0.50 documents per second, about 1,800 an hour)" in text
    assert "0.60 USD an hour" in text and "about 0.33 USD per 1,000 documents" in text
    assert "not a load benchmark" in text
    assert "not run" in cost_section(tmp_path / "none.json", costs)[0]


def test_the_price_file_names_the_currency_and_an_hourly_price():
    price = yaml.safe_load((ROOT / "configs" / "costs.yaml").read_text(encoding="utf-8"))["gpu"]
    assert price["currency"] == "USD" and price["per_hour"] > 0


def test_the_block_is_replaced_as_a_whole_and_a_second_run_changes_nothing():
    text = f"before\n{START}\nold\n{END}\nafter\n"
    once = update_document(text, "new")
    assert once == f"before\n{START}\nnew\n{END}\nafter\n"
    assert update_document(once, "new") == once


def test_a_readme_without_the_markers_is_an_error():
    with pytest.raises(ValueError, match="results:start"):
        update_document("no markers", "new")


def test_the_chart_is_a_png_of_the_two_dev_splits(tmp_path):
    rescored = ROOT / "results" / "round3" / "rescored"
    series = chart_series(rescored, "tax-booking", SYSTEM, 3)
    assert [item.split for item in series] == ["dev", "dev_real"]
    from adapt.config import load_platform

    path = tmp_path / "img" / "chart.png"
    write_chart(path, series, load_platform("tax-booking").config.gate)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 5000


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_the_make_target_passes_its_variables_on():
    def planned(*variables):
        return subprocess.run(
            ["make", "--dry-run", "--no-print-directory", "report", *variables],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    command = planned("SYSTEM=a", "BASELINE=b", "ROUND=3")
    assert "-m adapt.report" in command and "--system a" in command
    assert "--baseline b" in command and "--round 3" in command and "--check" not in command
    assert "--check" in planned("SYSTEM=a", "CHECK=1")

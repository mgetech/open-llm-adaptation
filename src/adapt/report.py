"""The results blocks of the README and of docs/results.md, and the coverage chart, written from
the saved results.

Run with `make report`. Every number comes from a file in `results/`, and a result that is not
there shows as "not run". In each document the block sits between two markers and is replaced as
a whole. `--check` writes nothing and fails if a block is not the one the results give.

The cost per 1,000 documents is not a benchmark. It comes from the time one evaluation run took
(`performance` in its results file) and the price in `configs/costs.yaml`, and the text says so.
"""

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from adapt.config import Gate, load_platform
from adapt.data.rounds import resolve_round
from adapt.eval.gate import Check, check_gate, load_results
from adapt.eval.rescore import rows_without_limit
from adapt.eval.runner import DATA_DIR, ROOT, load_documents, results_file
from adapt.eval.stage_gate import STAGE_SPLITS, checks, verdict
from adapt.eval.threshold import CurvePoint, coverage_curve, point_at

START = "<!-- results:start -->"
END = "<!-- results:end -->"
NOT_RUN = "not run"
COSTS = ROOT / "configs" / "costs.yaml"
CHART = Path("docs") / "img" / "coverage_accuracy.png"
README, RESULTS = "README.md", "docs/results.md"
VALIDATION_METRICS = (
    "automation_rate",
    "accuracy_on_automated",
    "critical_error_rate",
    "review_recall",
)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES_COLOURS = {"dev": "#2a78d6", "dev_real": "#eb6834"}
GOLD_SETS = ("gold", "gold_v2")  # the gate reads one of them; the claim is read with both


@dataclass(frozen=True)
class Series:
    split: str
    curve: list[CurvePoint]
    chosen: CurvePoint


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _number(value: float | None, places: int = 3) -> str:
    return NOT_RUN if value is None else f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _critical_count(results: Mapping[str, Any]) -> str:
    automated = [row for row in results["documents"] if not row["needs_review"]]
    critical = sum(row["severity"] == "critical" for row in automated)
    return f"{critical} of {len(automated)} automated"


def validation_section(rescored: Path, platform: str, system: str, gate: Gate) -> list[str]:
    saved = {
        split: _load(results_file(rescored, platform, system, split)) for split in STAGE_SPLITS
    }
    if not all(saved.values()):
        return [f"**Validation, dev and dev_real:** {NOT_RUN}."]
    threshold = saved["dev"]["manifest"]["threshold"]
    rows = []
    for name, metric, operator, limit in checks(gate):
        if metric not in VALIDATION_METRICS:
            continue
        cells = []
        for split in STAGE_SPLITS:
            value = saved[split]["metrics"][metric]["value"]
            places = 4 if metric == "critical_error_rate" else 3
            cell = f"{_number(value, places)} {verdict(value, operator, limit)}"
            if metric == "critical_error_rate":
                cell += f" ({_critical_count(saved[split])})"
            cells.append(cell)
        rows.append([name, f"{operator} {limit:g}", *cells])
    return [
        f"**Validation, not a test.** `{system}` on dev and `dev_real`, with the confidence "
        f"threshold {threshold:.4f} chosen on both. I chose the settings on these two sets, so "
        "they guide choices and are not a claim.",
        "",
        *_table(["check", "limit", *STAGE_SPLITS], rows),
    ]


def _check_cell(check: Check) -> str:
    if check.value is None:
        old = " saved with the old definition" if "old definition" in check.detail else ""
        return f"no value,{old} FAIL" if old else "no value FAIL"
    return f"{check.value:.4f} {'pass' if check.passed else 'FAIL'}"


def claim_section(
    eval_dir: Path, platform: str, gate: Gate, system: str, baseline: str | None
) -> list[str]:
    baseline_results = (
        _load(results_file(eval_dir, platform, baseline, "test_ood")) if baseline else None
    )
    gates: dict[str, list[Check]] = {}
    for gold in GOLD_SETS:
        results = load_results(eval_dir, platform, system, gold)
        if results:
            note = f"{NOT_RUN}: no baseline test_ood results"
            gates[gold] = check_gate(gate, results, baseline_results, note, gold)
    if not gates:
        return [f"**Held out, the claim run:** {NOT_RUN}."]
    passed = "; ".join(
        f"with `{gold}` {sum(c.passed for c in done)} of {len(done)} checks"
        for gold, done in gates.items()
    )
    every_check = list(zip(*gates.values(), strict=True))
    failed = [row for row in every_check if not all(check.passed for check in row)]
    lines = [
        f"**Held out: the claim run on the round 3 test sets.** The gate passed {passed}. "
        + ("It did not pass the gate." if failed else "It passed the gate.")
    ]
    if failed:
        lines += [
            "",
            *_table(
                ["failed check", "limit", *(f"gate with `{gold}`" for gold in gates)],
                [
                    [row[0].name, f"{row[0].operator} {row[0].limit:g}", *map(_check_cell, row)]
                    for row in failed
                ],
            ),
        ]
    return lines


def post_hoc_section(path: Path) -> list[str]:
    saved = _load(path)
    if saved is None:
        return [f"**Post hoc, on sets that were used before:** {NOT_RUN}."]
    splits = saved["splits"]
    names = list(splits)
    rows = [
        ["documents", *(str(splits[name]["documents"]) for name in names)],
        ["automation rate", *(_number(splits[n]["metrics"]["automation_rate"]) for n in names)],
        [
            "accuracy on automated documents",
            *(_number(splits[n]["metrics"]["accuracy_on_automated"]) for n in names),
        ],
        [
            "critical errors",
            *(
                f"{splits[n]['critical_errors']} of {splits[n]['automated']} automated"
                for n in names
            ),
        ],
        ["review recall", *(_number(splits[n]["metrics"]["review_recall"]) for n in names)],
    ]
    return [
        "**Post hoc, on sets that were used before, not held out.** The saved outputs of the same "
        "configuration scored again with every check of today.",
        "",
        *_table(["", *names], rows),
    ]


def training_section(results: Path) -> list[str]:
    rows = []
    for path in sorted(results.glob("round*/runs/*.json")):
        if "-steps" in path.name:
            continue  # the short runs that measured training speed
        run = json.loads(path.read_text(encoding="utf-8"))
        gpus = run.get("num_gpus")
        rows.append(
            [
                run["run_id"],
                path.parent.parent.name.removeprefix("round"),
                str(run["steps"]),
                "not stated" if gpus is None else str(gpus),
                f"{run['gpu_hours']:.1f}",
            ]
        )
    if not rows:
        return [f"**Training runs:** {NOT_RUN}."]
    return [
        "**Training runs**, with the number of GPUs each one used.",
        "",
        *_table(["run", "round", "steps", "GPUs", "GPU hours"], rows),
    ]


def azure_section(azure: Path) -> list[str]:
    lines = []
    for split in STAGE_SPLITS:
        saved = _load(azure / f"compare-{split}.json")
        if saved is None:
            continue
        lines.append(
            f"`{split}` {'passes' if saved['passed'] else 'fails'}, "
            f"{saved['identical_outputs']} of {saved['documents']} answers identical"
        )
    if not lines:
        return [f"**Azure T4 against Kaggle:** {NOT_RUN}."]
    return [
        "**Azure T4 against Kaggle**, the same adapter, prompt and threshold: "
        + "; ".join(lines)
        + ". Details are in `results/azure/README.md`."
    ]


def cost_section(azure_results: Path, costs: Path = COSTS) -> list[str]:
    saved = _load(azure_results)
    price = yaml.safe_load(costs.read_text(encoding="utf-8"))["gpu"] if costs.is_file() else None
    performance = saved and saved.get("performance")
    if not performance or not performance.get("documents_per_second") or not price:
        return [f"**Accuracy against cost per 1,000 documents:** {NOT_RUN}."]
    per_second = performance["documents_per_second"]
    per_hour = round(per_second * 3600, -2)
    cost = price["per_hour"] / (per_second * 3600) * 1000
    return [
        f"Accuracy against cost per 1,000 documents: {NOT_RUN}, because the load benchmark is "
        "future work. One figure exists as a by-product of an evaluation run. "
        f"The Azure evaluation processed {saved['manifest']['documents']} `{saved['split']}` "
        f"documents in {performance['wall_seconds']:.0f} seconds on one T4, with "
        f"{performance['workers']} requests in flight ({per_second:.2f} documents per second, "
        f"about {per_hour:,.0f} an hour). At {price['per_hour']:.2f} {price['currency']} an hour, "
        f"the on-demand price of the machine, that is about {cost:.2f} {price['currency']} per "
        "1,000 documents. It is not a load benchmark, and the price is for the VM alone."
    ]


def build_blocks(
    root: Path, platform: str, system: str, baseline: str | None, data_round: int
) -> dict[str, str]:
    """The text between the markers of each document, by the path of the document. The README
    keeps the validation table, and the other tables, the chart and the cost go to the results."""
    gate = load_platform(platform).config.gate
    results = root / "results"
    folder = results / f"round{data_round}"
    readme = [
        validation_section(folder / "rescored", platform, system, gate),
        [
            "The held-out result, the post-hoc re-score, the training runs, the run on Azure and "
            "the chart are in `docs/results.md`."
        ],
    ]
    details = [
        claim_section(folder / "eval", platform, gate, system, baseline),
        post_hoc_section(results / "post_hoc" / "gold_rescored.json"),
        training_section(results),
        azure_section(results / "azure"),
        [
            "![Automation rate and accuracy on automated documents at every confidence "
            f"threshold]({CHART.relative_to('docs').as_posix()})"
        ],
        cost_section(
            results_file(results / "azure" / "eval", platform, f"{system}-azure", "dev"),
            root / "configs" / "costs.yaml",
        ),
    ]
    return {
        document: "\n\n".join("\n".join(section) for section in sections)
        for document, sections in ((README, readme), (RESULTS, details))
    }


def update_document(text: str, block: str) -> str:
    start, end = text.find(START), text.find(END)
    if start < 0 or end < start:
        raise ValueError(f"the document needs the lines {START} and {END}")
    return f"{text[: start + len(START)]}\n{block}\n{text[end:]}"


def chart_series(rescored: Path, platform: str, system: str, data_round: int) -> list[Series]:
    loaded = load_platform(platform)
    series = []
    for split in STAGE_SPLITS:
        saved = _load(results_file(rescored, platform, system, split))
        if saved is None:
            continue
        documents = load_documents(platform, split, DATA_DIR, data_round)
        rows = rows_without_limit(saved, documents, loaded)
        series.append(
            Series(split, coverage_curve(rows), point_at(rows, saved["manifest"]["threshold"]))
        )
    return series


def write_chart(path: Path, series: Sequence[Series], gate: Gate) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    for item in series:
        points = sorted(
            (p.automation_rate, p.accuracy_on_automated)
            for p in item.curve
            if p.automation_rate is not None and p.accuracy_on_automated is not None
        )
        colour = SERIES_COLOURS[item.split]
        axes.plot(*zip(*points, strict=True), color=colour, linewidth=2, label=item.split)
        if item.chosen.automation_rate is not None and item.chosen.accuracy_on_automated:
            axes.plot(
                item.chosen.automation_rate,
                item.chosen.accuracy_on_automated,
                "o",
                color=colour,
                markersize=9,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
            )
    axes.axhline(gate.accuracy_automated_min, color=MUTED, linewidth=1, linestyle="--")
    axes.axvline(gate.automation_rate_min, color=MUTED, linewidth=1, linestyle="--")
    axes.text(0.01, gate.accuracy_automated_min + 0.004, "accuracy limit", color=MUTED, size=9)
    lowest = min(
        (p.accuracy_on_automated for item in series for p in item.curve if p.accuracy_on_automated),
        default=gate.accuracy_automated_min,
    )
    bottom = min(math.floor((lowest - 0.01) * 20) / 20, gate.accuracy_automated_min - 0.05)
    axes.text(
        gate.automation_rate_min + 0.01, bottom + 0.004, "automation limit", color=MUTED, size=9
    )
    axes.set_xlim(0, 1)
    axes.set_ylim(bottom, 1.01)
    axes.set_xlabel("automation rate", color=INK)
    axes.set_ylabel("accuracy on automated documents", color=INK)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    legend = axes.legend(frameon=False, loc="lower left", labelcolor=INK)
    legend.set_title("the dot marks the chosen threshold", prop={"size": 9})
    legend.get_title().set_color(MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the results blocks and the chart.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--system", required=True, help="the shipped configuration")
    parser.add_argument("--baseline", help="the few-shot system of the same size")
    parser.add_argument("--round", type=int, help="round of the results (default: the current one)")
    parser.add_argument("--root", type=Path, default=ROOT, help="the repository to read")
    parser.add_argument("--check", action="store_true", help="write nothing, fail on a difference")
    args = parser.parse_args(argv)
    try:
        data_round = resolve_round(args.round, DATA_DIR)
        blocks = build_blocks(args.root, args.platform, args.system, args.baseline, data_round)
        current = {name: (args.root / name).read_text(encoding="utf-8") for name in blocks}
        updated = {name: update_document(current[name], block) for name, block in blocks.items()}
    except (ValueError, OSError) as error:
        sys.exit(f"error: {error}")
    stale = [name for name in blocks if updated[name] != current[name]]
    if args.check:
        print(f"results out of date in {', '.join(stale)}" if stale else "results: up to date")
        return 1 if stale else 0
    folder = args.root / "results" / f"round{data_round}" / "rescored"
    series = chart_series(folder, args.platform, args.system, data_round)
    if series:
        write_chart(args.root / CHART, series, load_platform(args.platform).config.gate)
    for name, text in updated.items():
        (args.root / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {', '.join(updated)}" + (f" and {CHART.as_posix()}" if series else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

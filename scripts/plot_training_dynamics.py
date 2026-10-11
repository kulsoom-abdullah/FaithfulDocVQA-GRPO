"""Reproduce the public training-dynamics figures from compact public artifacts.

The plotting functions return Matplotlib ``Figure`` objects for notebook use.
The CLI saves one figure or the complete public set without rerunning OCR,
training, or inference.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.public_metrics import (  # noqa: E402
    aggregate_final_system_metrics,
    load_evaluation_ledger,
    load_public_jsonl,
    load_public_manifest,
)

ORIGINAL_SFT_STEPS = (214, 428, 642, 856, 1070, 1284)
BALANCED_SFT_STEPS = (324, 648, 972, 1296, 1620, 1944, 1947)
RUN4_VALIDATION_STEPS = tuple(range(10, 151, 10))
RUN4_SELECTED_STEPS = {"run-4a": 100, "run-4b": 150}

VALID_CITATION = "valid_citation"
HONEST_ABSTAIN = "honest_abstain"
FALSE_GROUNDING = "false_grounding"
FALSE_ABSTAIN = "false_abstain"
FABRICATION = "fabrication"
MALFORMED = "malformed"
CONTRADICTION = "contradiction"
ALLOWED_BUCKETS = {
    VALID_CITATION,
    HONEST_ABSTAIN,
    FALSE_GROUNDING,
    FALSE_ABSTAIN,
    FABRICATION,
    MALFORMED,
    CONTRADICTION,
    "uncited_answer",
}

RUN4_COLORS = {"run-4a": "#0072B2", "run-4b": "#D55E00"}
SFT_COLORS = {"original SFT": "#0072B2", "balanced SFT": "#009E73"}
BUCKET_COLORS = {
    HONEST_ABSTAIN: "#009E73",
    FALSE_GROUNDING: "#D55E00",
    VALID_CITATION: "#0072B2",
    FALSE_ABSTAIN: "#CC79A7",
    FABRICATION: "#E69F00",
    "format_failure": "#000000",
}
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
}


def _repo_root(repo_root: Path | None) -> Path:
    root = (repo_root or ROOT).resolve()
    sentinels = (root / "artifacts/public/manifest.json", root / "scripts/public_metrics.py")
    if not all(path.is_file() for path in sentinels):
        raise FileNotFoundError(f"not a public FaithfulDocVQA artifact tree: {root}")
    return root


def _public_dir(repo_root: Path) -> Path:
    return repo_root / "artifacts/public"


def _trailing_weighted_rate(
    numerators: np.ndarray, denominators: np.ndarray, window: int
) -> np.ndarray:
    """Return trailing weighted rates while preserving current-step population gaps."""

    result = np.full(len(numerators), np.nan, dtype=float)
    for index in range(len(numerators)):
        start = max(0, index - window + 1)
        denominator = denominators[start : index + 1].sum()
        if denominators[index] > 0 and denominator > 0:
            result[index] = numerators[start : index + 1].sum() / denominator
    return result


def _padded_rate_limits(
    curves: dict[str, list[dict[str, float | bool]]],
    field: str,
    padding: float,
) -> tuple[float, float]:
    """Return data-derived rate limits with bounded fractional padding."""

    values = [float(point[field]) for points in curves.values() for point in points]
    return max(0.0, min(values) - padding), min(1.0, max(values) + padding)


def _load_reward_logs(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    public_dir = _public_dir(repo_root)
    manifest = load_public_manifest(public_dir)
    rows = load_public_jsonl(
        public_dir,
        "training/run4_reward_dynamics.jsonl",
        manifest=manifest,
    )
    if len(rows) != 300:
        raise ValueError(f"expected 300 run-4 reward rows, got {len(rows)}")
    logs: dict[str, list[dict[str, Any]]] = {}
    for system_id, run in (("run4a", "run-4a"), ("run4b", "run-4b")):
        run_rows = [row for row in rows if row["system_id"] == system_id]
        if [row["optimizer_step"] for row in run_rows] != list(range(1, 151)):
            raise ValueError(f"reward steps are not exactly 1..150 for {run}")
        if any(row["completion_count"] != 32 for row in run_rows):
            raise ValueError(f"reward rows do not contain 32 completions per step for {run}")
        selected = [row["optimizer_step"] for row in run_rows if row["selected"]]
        if selected != [RUN4_SELECTED_STEPS[run]]:
            raise ValueError(f"unexpected selected checkpoint in reward summary for {run}")
        running_ema: float | None = None
        transformed: list[dict[str, Any]] = []
        for row in run_rows:
            reward = float(row["mean_reward"])
            running_ema = reward if running_ema is None else 0.9 * running_ema + 0.1 * reward
            reward_ema = float(row["reward_ema_alpha_0_1"])
            if not math.isclose(reward_ema, round(running_ema, 4), abs_tol=1e-12):
                raise ValueError(f"reward EMA is not the committed alpha=0.1 recurrence for {run}")
            transformed.append(
                {"step": int(row["optimizer_step"]), "reward": reward, "reward_ema": reward_ema}
            )
        logs[run] = transformed
    return logs


def _load_outcome_summaries(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    public_dir = _public_dir(repo_root)
    manifest = load_public_manifest(public_dir)
    rows = load_public_jsonl(
        public_dir,
        "training/run4_outcome_dynamics.jsonl",
        manifest=manifest,
    )
    if len(rows) != 600:
        raise ValueError(f"expected 600 run-4 outcome rows, got {len(rows)}")
    summaries: dict[str, list[dict[str, Any]]] = {}
    for system_id, run in (("run4a", "run-4a"), ("run4b", "run-4b")):
        run_rows = [row for row in rows if row["system_id"] == system_id]
        if len(run_rows) != 300:
            raise ValueError(f"expected 300 outcome rows for {run}")
        by_key = {(row["optimizer_step"], row["target_type"]): row for row in run_rows}
        expected_keys = {
            (step, target_type)
            for step in range(1, 151)
            for target_type in ("abstention_target", "answerable")
        }
        if set(by_key) != expected_keys:
            raise ValueError(f"outcome rows do not cover 150 steps and two target types for {run}")
        for step in range(1, 151):
            step_total = 0
            for target_type in ("abstention_target", "answerable"):
                row = by_key[(step, target_type)]
                if set(row["bucket_counts"]) != ALLOWED_BUCKETS:
                    raise ValueError(f"unexpected outcome bucket schema for {run} step {step}")
                if sum(row["bucket_counts"].values()) != row["completion_count"]:
                    raise ValueError(f"outcome buckets do not partition {run} step {step}")
                step_total += int(row["completion_count"])
            if step_total != 32:
                raise ValueError(f"outcome rows do not total 32 completions for {run} step {step}")
        summaries[run] = run_rows
    return summaries


def _build_run4_validation_curves(repo_root: Path) -> dict[str, list[dict[str, float]]]:
    public_dir = _public_dir(repo_root)
    manifest = load_public_manifest(public_dir)
    rows = load_public_jsonl(
        public_dir,
        "training/run4_validation.jsonl",
        manifest=manifest,
    )
    if len(rows) != 30:
        raise ValueError(f"expected 30 run-4 validation rows, got {len(rows)}")
    curves: dict[str, list[dict[str, float]]] = {}
    for system_id, run in (("run4a", "run-4a"), ("run4b", "run-4b")):
        run_rows = [row for row in rows if row["system_id"] == system_id]
        if [row["checkpoint"] for row in run_rows] != list(RUN4_VALIDATION_STEPS):
            raise ValueError(f"unexpected run-4 validation checkpoint set for {run}")
        if [row["checkpoint"] for row in run_rows if row["selected"]] != [
            RUN4_SELECTED_STEPS[run]
        ]:
            raise ValueError(f"unexpected selected validation checkpoint for {run}")
        points: list[dict[str, float]] = []
        for row in run_rows:
            if (
                row["total_denominator"] != 112
                or row["abstention_denominator"] != 64
                or row["answerable_denominator"] != 48
            ):
                raise ValueError(f"wrong run-4 validation population for {run}")
            points.append(
                {
                    "step": float(row["checkpoint"]),
                    "honest_count": float(row["honest_refusal_count"]),
                    "false_count": float(row["false_refusal_count"]),
                    "honest_refusal": row["honest_refusal_count"] / 64,
                    "false_refusal": row["false_refusal_count"] / 48,
                    "citation_f1": float(row["citation_micro_f1"]),
                    "eligible_count": float(row["citation_eligible_count"]),
                    "coverage": row["citation_eligible_count"] / 48,
                }
            )
        curves[run] = points
    return curves


def _build_sft_validation_curves(
    repo_root: Path,
) -> dict[str, list[dict[str, float | bool]]]:
    public_dir = _public_dir(repo_root)
    manifest = load_public_manifest(public_dir)
    rows = load_public_jsonl(
        public_dir,
        "training/sft_validation.jsonl",
        manifest=manifest,
    )
    if len(rows) != 13:
        raise ValueError(f"expected 13 SFT validation rows, got {len(rows)}")
    series = {
        "original SFT": ("original_sft", ORIGINAL_SFT_STEPS, 642),
        "balanced SFT": ("balanced_sft", BALANCED_SFT_STEPS, 1296),
    }
    curves: dict[str, list[dict[str, float | bool]]] = {}
    for display_name, (system_id, expected_steps, selected_step) in series.items():
        run_rows = [row for row in rows if row["system_id"] == system_id]
        if tuple(row["checkpoint"] for row in run_rows) != expected_steps:
            raise ValueError(f"unexpected checkpoint set for {display_name}")
        if [row["checkpoint"] for row in run_rows if row["selected"]] != [selected_step]:
            raise ValueError(f"unexpected selected checkpoint for {display_name}")
        points: list[dict[str, float | bool]] = []
        for row in run_rows:
            if row["abstention_denominator"] != 46 or row["answerable_denominator"] != 707:
                raise ValueError(f"wrong SFT validation population for {display_name}")
            points.append(
                {
                    "step": float(row["checkpoint"]),
                    "epoch": float(row["normalized_epoch"]),
                    "honest_refusal": row["honest_refusal_count"] / 46,
                    "false_refusal": row["false_refusal_count"] / 707,
                    "citation_f1": float(row["citation_micro_f1"]),
                    "coverage": row["citation_eligible_count"] / 707,
                    "selected": bool(row["selected"]),
                }
            )
        curves[display_name] = points
    return curves


def _build_final_operating_points(repo_root: Path) -> list[dict[str, float | int | str]]:
    public_dir = _public_dir(repo_root)
    manifest = load_public_manifest(public_dir)
    ledger = load_evaluation_ledger(public_dir, manifest=manifest)
    metrics = aggregate_final_system_metrics(ledger)
    expected = manifest.get("expected_headline_metrics")
    if metrics != expected:
        raise ValueError("public ledger final metrics disagree with the public manifest")
    systems = (
        ("Base", "base"),
        ("SFT seed", "sft_seed"),
        ("Run-1", "run1"),
        ("Run-2", "run2"),
        ("Balanced SFT", "balanced_sft"),
        ("Run-4a", "run4a"),
        ("Run-4b", "run4b"),
    )
    points: list[dict[str, float | int | str]] = []
    for name, system_id in systems:
        system = metrics[system_id]
        false_metric = system["natural_answerable_false_refusal"]
        clean_metric = system["clean_manufactured_honest_refusal"]
        natural_metric = system["natural_ocr_miss_honest_refusal"]
        if (
            false_metric["denominator"] != 679
            or clean_metric["denominator"] != 235
            or natural_metric["denominator"] != 77
        ):
            raise ValueError(f"wrong final evaluation denominators for {name}")
        points.append(
            {
                "name": name,
                "false_count": false_metric["count"],
                "clean_count": clean_metric["count"],
                "natural_count": natural_metric["count"],
                "false_rate": false_metric["count"] / 679,
                "clean_rate": clean_metric["count"] / 235,
            }
        )
    return points


def plot_sft_checkpoint_behavior(repo_root: Path | None = None) -> Figure:
    """Plot original and balanced SFT behavior across validation checkpoints."""

    root = _repo_root(repo_root)
    curves = _build_sft_validation_curves(root)
    metrics = (
        ("honest_refusal", "Natural OCR-miss honest refusal · n=46"),
        ("false_refusal", "False refusal · n=707"),
        ("citation_f1", "Conditional citation-label F1"),
        (
            "coverage",
            "Fraction of 707 answerable rows eligible for conditional citation-label F1",
        ),
    )
    y_limits = {
        "honest_refusal": _padded_rate_limits(curves, "honest_refusal", 0.05),
        "false_refusal": _padded_rate_limits(curves, "false_refusal", 0.015),
        "citation_f1": _padded_rate_limits(curves, "citation_f1", 0.01),
        "coverage": _padded_rate_limits(curves, "coverage", 0.01),
    }
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(11.2, 7.8),
            sharex=True,
            gridspec_kw={"height_ratios": (1.35, 1.0)},
        )
        for axis, (field, title) in zip(axes.flat, metrics, strict=True):
            for run, marker, linestyle in (
                ("original SFT", "o", "-"),
                ("balanced SFT", "s", "--"),
            ):
                points = curves[run]
                epochs = np.asarray([float(point["epoch"]) for point in points])
                values = np.asarray([float(point[field]) for point in points])
                color = SFT_COLORS[run]
                axis.plot(
                    epochs,
                    values,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=run,
                )
                selected = next(point for point in points if point["selected"])
                axis.scatter(
                    [float(selected["epoch"])],
                    [float(selected[field])],
                    s=75,
                    marker=marker,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=2.0,
                    zorder=5,
                )
                label = "ck-642" if run == "original SFT" else "ck-1296"
                offset = (6, -17) if run == "original SFT" else (6, 8)
                axis.annotate(
                    label,
                    (float(selected["epoch"]), float(selected[field])),
                    xytext=offset,
                    textcoords="offset points",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                )
            axis.set_title(title)
            axis.set_ylim(*y_limits[field])
            axis.set_xlim(0.45, 3.05)
            axis.legend(loc="best")
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        for axis in axes[1, :]:
            axis.set_xlabel("Normalized training epoch")
        figure.suptitle(
            "SFT validation behavior across checkpoints",
            fontsize=14,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.008,
            "The balanced run adds 1,764 manufactured abstention examples, so normalized epoch aligns training progress despite different optimizer-step counts.\n"
            "Open markers identify selected checkpoints. Panel y-ranges differ to make checkpoint-level variation visible.",
            ha="center",
            fontsize=8.8,
        )
        figure.tight_layout(rect=(0, 0.065, 1, 0.95))
    return figure


def plot_run4_reward_dynamics(repo_root: Path | None = None) -> Figure:
    """Plot raw mean reward and committed EMA for run-4a and run-4b."""

    root = _repo_root(repo_root)
    logs = _load_reward_logs(root)
    all_rewards = [float(row["reward"]) for rows in logs.values() for row in rows]
    lower, upper = min(all_rewards), max(all_rewards)
    margin = 0.08 * (upper - lower)

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True, sharey=True)
        for axis, run in zip(axes, ("run-4a", "run-4b"), strict=True):
            rows = logs[run]
            steps = np.asarray([int(row["step"]) for row in rows])
            raw = np.asarray([float(row["reward"]) for row in rows])
            ema = np.asarray([float(row["reward_ema"]) for row in rows])
            color = RUN4_COLORS[run]
            axis.plot(
                steps,
                raw,
                color=color,
                alpha=0.25,
                linewidth=1.0,
                label="per-step mean · 32 completions",
            )
            axis.plot(
                steps,
                ema,
                color=color,
                linewidth=2.3,
                label="exponential moving average (α=0.1; smoothed trend)",
            )
            checkpoint = RUN4_SELECTED_STEPS[run]
            axis.axvline(checkpoint, color="#333333", linestyle="--", linewidth=1.1)
            axis.scatter(
                [checkpoint],
                [ema[checkpoint - 1]],
                color="#333333",
                marker="D" if run == "run-4a" else "s",
                s=32,
                zorder=4,
            )
            intervention = (
                "no false-refusal penalty"
                if run == "run-4a"
                else "−0.2 false-refusal penalty"
            )
            axis.set_title(f"{run} · {intervention} · selected ck-{checkpoint}")
            axis.set_xlim(1, 155)
            axis.set_ylim(lower - margin, upper + margin)
            axis.set_xlabel("Optimizer step")
            axis.legend(loc="lower right")
        axes[0].set_ylabel("Mean scalar reward")
        figure.suptitle("Mean GRPO reward over training", fontsize=14, fontweight="bold")
        figure.text(
            0.5,
            0.012,
            "Each optimizer step contains 4 training examples × 8 sampled completions = 32 completions.\n"
            "Run-4b adds a false-refusal penalty, so its reward scale differs slightly from run-4a; read each panel as a within-run trajectory, not a claim that higher absolute reward means better behavior.\n"
            "Selected checkpoints came from the fixed 112-example validation set evaluated every ten steps, not from the maximum training reward.",
            ha="center",
            fontsize=8.6,
        )
        figure.tight_layout(rect=(0, 0.13, 1, 0.93))
    return figure


def plot_run4_outcome_dynamics(repo_root: Path | None = None) -> Figure:
    """Plot run-4 sampled outcome composition using within-population rates."""

    root = _repo_root(repo_root)
    summaries = _load_outcome_summaries(root)
    populations = (
        (
            True,
            "Abstention-target training examples",
            (HONEST_ABSTAIN, FALSE_GROUNDING, FABRICATION, "format_failure"),
        ),
        (
            False,
            "Answerable training examples",
            (VALID_CITATION, FALSE_ABSTAIN, FABRICATION, "format_failure"),
        ),
    )
    labels = {
        HONEST_ABSTAIN: "honest refusal",
        FALSE_GROUNDING: "false grounding",
        VALID_CITATION: "structurally valid citation claim",
        FALSE_ABSTAIN: "false refusal",
        FABRICATION: "fabrication",
        "format_failure": "malformed / contradiction",
    }

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharex=True, sharey=True)
        for row_index, (is_abstain, population_name, buckets) in enumerate(populations):
            for column, run in enumerate(("run-4a", "run-4b")):
                axis = axes[row_index, column]
                target_type = "abstention_target" if is_abstain else "answerable"
                by_step = {
                    int(item["optimizer_step"]): item
                    for item in summaries[run]
                    if item["target_type"] == target_type
                }
                denominators = np.asarray(
                    [by_step[step]["completion_count"] for step in range(1, 151)],
                    dtype=float,
                )
                for bucket in buckets:
                    bucket_set = (
                        {MALFORMED, CONTRADICTION}
                        if bucket == "format_failure"
                        else {bucket}
                    )
                    numerators = np.asarray(
                        [
                            sum(by_step[step]["bucket_counts"][name] for name in bucket_set)
                            for step in range(1, 151)
                        ],
                        dtype=float,
                    )
                    if numerators.sum() == 0:
                        continue
                    raw = np.divide(
                        numerators,
                        denominators,
                        out=np.full(150, np.nan, dtype=float),
                        where=denominators > 0,
                    )
                    smooth = _trailing_weighted_rate(numerators, denominators, 10)
                    steps = np.arange(1, 151)
                    color = BUCKET_COLORS[bucket]
                    linestyle = "--" if bucket in {FABRICATION, "format_failure"} else "-"
                    axis.plot(steps, raw, color=color, alpha=0.16, linewidth=0.9)
                    axis.plot(
                        steps,
                        smooth,
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.0,
                        label=labels[bucket],
                    )
                checkpoint = RUN4_SELECTED_STEPS[run]
                axis.axvline(checkpoint, color="#555555", linestyle="--", linewidth=1.0)
                intervention = (
                    "no false-refusal penalty"
                    if run == "run-4a"
                    else "−0.2 false-refusal penalty"
                )
                axis.set_title(f"{run} · {population_name}\n{intervention}")
                axis.set_ylim(-0.03, 1.03)
                axis.set_xlim(1, 155)
                if row_index == 1:
                    axis.set_xlabel("Optimizer step")
                axis.legend(loc="center right")
        figure.suptitle("Run-4 sampled outcome composition", fontsize=14, fontweight="bold")
        figure.supylabel(
            "Fraction of sampled completions within target type",
            x=0.012,
            fontsize=10,
        )
        figure.text(
            0.5,
            0.925,
            "Each optimizer step contains 4 examples × 8 completions; each panel pools only completions from the named target type.",
            ha="center",
            fontsize=9,
        )
        figure.text(
            0.5,
            0.01,
            "Faint lines are raw per-step fractions; bold lines are trailing 10-step weighted averages. Steps with no example of a target type are undefined and left blank, not treated as zero.\n"
            "Legends show only outcome categories observed in that panel.",
            ha="center",
            fontsize=8.7,
        )
        figure.tight_layout(rect=(0.045, 0.065, 1, 0.9))
    return figure


def plot_run4_validation_dynamics(repo_root: Path | None = None) -> Figure:
    """Plot held-out run-4 validation behavior at checkpoints 10 through 150."""

    root = _repo_root(repo_root)
    curves = _build_run4_validation_curves(root)
    metrics = (
        (
            "honest_count",
            "Honest refusals out of 64 abstention-target validation examples",
        ),
        (
            "false_count",
            "False refusals out of 48 answerable validation examples",
        ),
    )
    run_styles = {
        "run-4a": {"marker": "D", "linestyle": "-"},
        "run-4b": {"marker": "s", "linestyle": "--"},
    }

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), sharex=True)
        for axis, (field, title) in zip(axes, metrics, strict=True):
            all_values: list[float] = []
            for run in ("run-4a", "run-4b"):
                points = curves[run]
                steps = np.asarray([int(point["step"]) for point in points])
                values = np.asarray([float(point[field]) for point in points])
                all_values.extend(values.tolist())
                color = RUN4_COLORS[run]
                style = run_styles[run]
                axis.plot(
                    steps,
                    values,
                    color=color,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markersize=4,
                    linewidth=1.8,
                    label=f"{run} — selected ck-{RUN4_SELECTED_STEPS[run]}",
                )
                checkpoint = RUN4_SELECTED_STEPS[run]
                selected_value = values[list(steps).index(checkpoint)]
                axis.scatter(
                    [checkpoint],
                    [selected_value],
                    marker=style["marker"],
                    s=72,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=2.0,
                    zorder=5,
                )
            axis.set_title(title)
            lower = math.floor(min(all_values)) - 1
            upper = math.ceil(max(all_values)) + 1
            axis.set_ylim(lower, upper)
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_xlim(8, 157)
            axis.set_xticks((10, 30, 50, 70, 90, 110, 130, 150))
            axis.legend(loc="best")
        for axis in axes:
            axis.set_xlabel("Checkpoint step")
        figure.supylabel("Count of validation examples", x=0.012, fontsize=10)
        figure.suptitle(
            "Run-4 checkpoint selection on the same 112 validation examples",
            fontsize=14,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.012,
            "The same 64 abstention-target and 48 answerable validation examples were evaluated every ten steps; no test results are used.\n"
            "Run-4a ck-100 resolved a validation tie using the documented earliest-step rule. Run-4b ck-150 uniquely maximized honest refusal.",
            ha="center",
            fontsize=8.7,
        )
        figure.tight_layout(rect=(0.045, 0.09, 1, 0.93))
    return figure


def plot_final_operating_points(repo_root: Path | None = None) -> Figure:
    """Plot final selected-checkpoint test operating points for seven systems."""

    root = _repo_root(repo_root)
    points = _build_final_operating_points(root)
    colors = {
        "Base": "#777777",
        "SFT seed": "#8ECAE6",
        "Run-1": "#219EBC",
        "Run-2": "#126782",
        "Balanced SFT": "#95D5B2",
        "Run-4a": "#2D9D78",
        "Run-4b": "#12664F",
    }
    offsets = {
        "Base": (-8, 12),
        "SFT seed": (8, 10),
        "Run-1": (8, -30),
        "Run-2": (8, -42),
        "Balanced SFT": (-118, 25),
        "Run-4a": (12, -42),
        "Run-4b": (-132, -48),
    }

    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(9.8, 6.8))
        for point in points:
            name = str(point["name"])
            x = float(point["false_rate"])
            y = float(point["clean_rate"])
            axis.scatter(
                x,
                y,
                color=colors[name],
                marker="o",
                s=82,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            axis.annotate(
                f"{name}\nnatural OCR-miss {point['natural_count']}/77",
                (x, y),
                xytext=offsets[name],
                textcoords="offset points",
                fontsize=9,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#777777",
                    "alpha": 0.75,
                    "linewidth": 0.65,
                },
            )
        axis.set_xlabel("False refusal on answerable test rows · lower is better (n=679)")
        axis.set_ylabel(
            "Honest refusal on clean manufactured test rows · higher is better (n=235)"
        )
        axis.set_xlim(-0.005, 0.19)
        axis.set_ylim(-0.01, 1.01)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_title(
            "Final refusal trade-offs across evaluated systems",
            fontsize=14,
            fontweight="bold",
            pad=16,
        )
        figure.text(
            0.5,
            0.012,
            "Upper-left is preferred: more correct refusal after evidence removal and fewer incorrect refusals when evidence is present.\n"
            "Labels report honest refusal on the separate 77-row natural OCR-miss test. No evaluated system is best on all three measures.",
            ha="center",
            fontsize=8.7,
        )
        figure.tight_layout(rect=(0, 0.075, 1, 1))
    return figure


FIGURE_BUILDERS: dict[str, tuple[str, Callable[[Path | None], Figure]]] = {
    "sft_checkpoint_behavior": (
        "sft_checkpoint_behavior.png",
        plot_sft_checkpoint_behavior,
    ),
    "run4_reward_dynamics": (
        "run4_reward_dynamics.png",
        plot_run4_reward_dynamics,
    ),
    "run4_outcome_dynamics": (
        "run4_outcome_dynamics.png",
        plot_run4_outcome_dynamics,
    ),
    "run4_validation_dynamics": (
        "run4_validation_dynamics.png",
        plot_run4_validation_dynamics,
    ),
    "final_operating_points": (
        "final_operating_points.png",
        plot_final_operating_points,
    ),
}


def save_figures(figure_name: str, out_dir: Path, repo_root: Path | None = None) -> list[Path]:
    """Build and save one named figure or the full approved public set."""

    root = _repo_root(repo_root)
    if figure_name == "all":
        selected = list(FIGURE_BUILDERS.items())
    else:
        selected = [(figure_name, FIGURE_BUILDERS[figure_name])]
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, (filename, builder) in selected:
        figure = builder(root)
        output_path = out_dir / filename
        figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        saved.append(output_path)
        print(f"saved {name}: {output_path}")
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("all", *FIGURE_BUILDERS),
        default="all",
        help="figure to render (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/figures",
        help="directory for rendered PNG files",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="repository root containing artifacts/public and scripts/",
    )
    return parser.parse_args()


def main() -> None:
    """Render the requested approved figure set from repository artifacts."""

    args = _parse_args()
    save_figures(args.figure, args.out_dir, args.repo_root)


if __name__ == "__main__":
    main()

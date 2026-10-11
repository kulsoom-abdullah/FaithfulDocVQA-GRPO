"""CPU tests for public-artifact-only training-dynamics plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from scripts.plot_training_dynamics import (
    BALANCED_SFT_STEPS,
    ORIGINAL_SFT_STEPS,
    RUN4_VALIDATION_STEPS,
    _build_final_operating_points,
    _build_run4_validation_curves,
    _build_sft_validation_curves,
    _load_outcome_summaries,
    _load_reward_logs,
    plot_final_operating_points,
    plot_run4_outcome_dynamics,
    plot_run4_reward_dynamics,
    plot_run4_validation_dynamics,
    plot_sft_checkpoint_behavior,
)
from scripts.public_metrics import (
    FALSE_ABSTAIN,
    HONEST_ABSTAIN,
    aggregate_final_system_metrics,
    load_evaluation_ledger,
    load_public_manifest,
    paired_comparison,
)

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "artifacts/public"


def test_public_only_loading_and_final_metrics() -> None:
    manifest = load_public_manifest(PUBLIC_DIR)
    ledger = load_evaluation_ledger(PUBLIC_DIR, manifest=manifest)
    metrics = aggregate_final_system_metrics(ledger)
    assert metrics == manifest["expected_headline_metrics"]
    expected_counts = {
        "base": (4, 8, 14, 638),
        "sft_seed": (68, 62, 26, 650),
        "run1": (62, 59, 23, 655),
        "run2": (204, 77, 120, 557),
        "balanced_sft": (212, 71, 78, 601),
        "run4a": (217, 76, 99, 580),
        "run4b": (217, 74, 91, 588),
    }
    for system_id, expected in expected_counts.items():
        system = metrics[system_id]
        assert (
            system["clean_manufactured_honest_refusal"]["count"],
            system["natural_ocr_miss_honest_refusal"]["count"],
            system["natural_answerable_false_refusal"]["count"],
            system["conditional_citation_label_f1"]["eligible_count"],
        ) == expected


def test_exact_public_training_series() -> None:
    sft = _build_sft_validation_curves(ROOT)
    assert tuple(int(row["step"]) for row in sft["original SFT"]) == ORIGINAL_SFT_STEPS
    assert tuple(int(row["step"]) for row in sft["balanced SFT"]) == BALANCED_SFT_STEPS
    assert [int(row["step"]) for row in sft["original SFT"] if row["selected"]] == [642]
    assert [int(row["step"]) for row in sft["balanced SFT"] if row["selected"]] == [1296]

    reward = _load_reward_logs(ROOT)
    for run, selected in (("run-4a", 100), ("run-4b", 150)):
        assert len(reward[run]) == 150
        assert [row["step"] for row in reward[run]] == list(range(1, 151))
        assert selected in {row["step"] for row in reward[run]}

    outcomes = _load_outcome_summaries(ROOT)
    for run in ("run-4a", "run-4b"):
        assert len(outcomes[run]) == 300
        for target_type in ("abstention_target", "answerable"):
            rows = [row for row in outcomes[run] if row["target_type"] == target_type]
            assert [row["optimizer_step"] for row in rows] == list(range(1, 151))
        totals = {
            step: sum(
                row["completion_count"]
                for row in outcomes[run]
                if row["optimizer_step"] == step
            )
            for step in range(1, 151)
        }
        assert totals == {step: 32 for step in range(1, 151)}

    validation = _build_run4_validation_curves(ROOT)
    for run, selected in (("run-4a", 100), ("run-4b", 150)):
        assert tuple(int(row["step"]) for row in validation[run]) == RUN4_VALIDATION_STEPS
        selected_point = next(row for row in validation[run] if int(row["step"]) == selected)
        assert (selected_point["honest_count"], selected_point["false_count"]) == (60.0, 2.0)


@pytest.mark.parametrize(
    ("before", "after", "population", "bucket", "lower_is_better", "expected"),
    (
        ("sft_seed", "run1", "manufactured_clean", HONEST_ABSTAIN, False, (1, 7, 0.0703125)),
        (
            "sft_seed",
            "run2",
            "manufactured_clean",
            HONEST_ABSTAIN,
            False,
            (136, 0, 2.2958874039497803e-41),
        ),
        (
            "sft_seed",
            "balanced_sft",
            "manufactured_clean",
            HONEST_ABSTAIN,
            False,
            (144, 0, 8.96831017167883e-44),
        ),
        (
            "run2",
            "balanced_sft",
            "manufactured_clean",
            HONEST_ABSTAIN,
            False,
            (14, 6, 0.11531829833984375),
        ),
        (
            "balanced_sft",
            "run4a",
            "manufactured_clean",
            HONEST_ABSTAIN,
            False,
            (5, 0, 0.0625),
        ),
        (
            "balanced_sft",
            "run4b",
            "manufactured_clean",
            HONEST_ABSTAIN,
            False,
            (6, 1, 0.125),
        ),
        ("run4a", "run4b", "manufactured_clean", HONEST_ABSTAIN, False, (1, 1, 1.0)),
        ("run4a", "run4b", "natural_ocr_miss", HONEST_ABSTAIN, False, (0, 2, 0.5)),
        (
            "run4a",
            "run4b",
            "natural_menu_answerable",
            FALSE_ABSTAIN,
            True,
            (10, 2, 0.03857421875),
        ),
    ),
)
def test_notebook_comparison_coverage(
    before: str,
    after: str,
    population: str,
    bucket: str,
    lower_is_better: bool,
    expected: tuple[int, int, float],
) -> None:
    ledger = load_evaluation_ledger(PUBLIC_DIR)
    result = paired_comparison(
        ledger,
        before=before,
        after=after,
        population=population,
        endpoint_bucket=bucket,
        lower_is_better=lower_is_better,
    )
    assert (result["improved"], result["declined"]) == expected[:2]
    assert result["exact_two_sided_p"] == pytest.approx(expected[2])


def test_full_manufactured_run1_to_run2_comparison() -> None:
    manifest = load_public_manifest(PUBLIC_DIR)
    ledger = load_evaluation_ledger(PUBLIC_DIR)
    result = paired_comparison(
        ledger,
        before="run1",
        after="run2",
        population="manufactured_all",
        endpoint_bucket=HONEST_ABSTAIN,
    )
    assert result["denominator"] == 250
    assert (result["improved"], result["declined"]) == (144, 0)
    assert result["exact_two_sided_p"] == pytest.approx(8.96831017167883e-44)
    manifest_result = manifest["secondary_comparisons"][
        "run1_to_run2_full_manufactured"
    ]
    assert manifest_result["used_in_public_notebook"] is False
    assert manifest_result["focused_review_exclusions_included"] == 15
    for key, value in result.items():
        assert manifest_result[key] == value


def test_final_operating_points_use_all_seven_systems() -> None:
    points = _build_final_operating_points(ROOT)
    assert [point["name"] for point in points] == [
        "Base",
        "SFT seed",
        "Run-1",
        "Run-2",
        "Balanced SFT",
        "Run-4a",
        "Run-4b",
    ]
    assert [(point["false_count"], point["clean_count"], point["natural_count"]) for point in points] == [
        (14, 4, 8),
        (26, 68, 62),
        (23, 62, 59),
        (120, 204, 77),
        (78, 212, 71),
        (99, 217, 76),
        (91, 217, 74),
    ]


@pytest.mark.parametrize(
    "builder",
    (
        plot_sft_checkpoint_behavior,
        plot_run4_reward_dynamics,
        plot_run4_outcome_dynamics,
        plot_run4_validation_dynamics,
        plot_final_operating_points,
    ),
)
def test_all_public_figures_construct_on_cpu(builder) -> None:
    figure = builder(ROOT)
    try:
        assert isinstance(figure, Figure)
        assert figure.axes
    finally:
        plt.close(figure)


def test_plotting_source_has_no_raw_or_private_path_dependencies() -> None:
    source = (ROOT / "scripts/plot_training_dynamics.py").read_text(encoding="utf-8")
    for forbidden in (
        "data/master.jsonl",
        "data/checkpoints/eval",
        "reward_members.jsonl",
        "manufactured_test_abstains.jsonl",
        "docs/analysis/ARTIFACTS.sha256",
    ):
        assert forbidden not in source

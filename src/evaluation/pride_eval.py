from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from src import OPTION_LABELS
from src.pride.debias import apply_pride_correction
from src.pride.prior_estimation import estimate_global_prior, estimate_sample_prior


def apply_prior_to_records(
    baseline_rows: Sequence[dict[str, Any]],
    prior: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> list[dict[str, Any]]:
    output = []
    for row in baseline_rows:
        observed = [float(row[f"prob_{label}"]) for label in OPTION_LABELS]
        corrected = apply_pride_correction(observed, prior, epsilon=epsilon)
        pride_index = int(corrected.argmax())
        updated = dict(row)
        updated["original_prediction"] = row["predicted_label"]
        updated["pride_prediction"] = OPTION_LABELS[pride_index]
        updated["predicted_label"] = OPTION_LABELS[pride_index]
        updated["correct"] = OPTION_LABELS[pride_index] == row["gold_label"]
        for index, label in enumerate(OPTION_LABELS):
            updated[f"prior_{label}"] = float(prior[index])
            updated[f"corrected_prob_{label}"] = float(corrected[index])
        output.append(updated)
    return output


def split_estimation_and_remaining(
    pride_rows: Sequence[dict[str, Any]],
    cyclic_rows: Sequence[dict[str, Any]],
    estimation_ids: Sequence[str],
    *,
    transferred: bool = False,
) -> list[dict[str, Any]]:
    """Apply Algorithm 1's D_e vs D_r treatment and label each record explicitly.

    - Estimation set D_e: samples in ``estimation_ids``. Under the cyclic policy they
      receive the permutation-based (cyclic) debiased prediction from ``cyclic_rows``.
    - Remaining set D_r: everything else keeps the Equation-8 global-prior correction
      already present in ``pride_rows``.

    Every returned record carries ``sample_group`` ('D_e'/'D_r') and ``debiasing_method``.
    """
    cyclic = {str(row["sample_id"]): row for row in cyclic_rows}
    estimation = {str(sample_id) for sample_id in estimation_ids}
    output: list[dict[str, Any]] = []
    for row in pride_rows:
        sample_id = str(row["sample_id"])
        updated = dict(row)
        updated["sample_group"] = "D_e" if sample_id in estimation else "D_r"
        replacement = cyclic.get(sample_id)
        if replacement is not None:
            updated["debiasing_method"] = "cyclic_estimation_sample"
            updated["pride_prediction"] = replacement["predicted_label"]
            updated["predicted_label"] = replacement["predicted_label"]
            updated["correct"] = bool(replacement["correct"])
            for index, label in enumerate(OPTION_LABELS):
                updated[f"corrected_prob_{label}"] = replacement[f"cyclic_prob_content_{label}"]
        else:
            updated["debiasing_method"] = (
                "pride_transferred_global_prior" if transferred else "pride_global_prior"
            )
        output.append(updated)
    return output


def estimate_prior_from_permutation_records(
    rows: Sequence[dict[str, Any]],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Estimate PriDe priors without reading any gold-label field."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    sample_prior_rows = []
    sample_priors = []
    for sample_id, sample_rows in sorted(grouped.items()):
        ordered = sorted(sample_rows, key=lambda row: int(row["permutation_id"]))
        if [int(row["permutation_id"]) for row in ordered] != [0, 1, 2, 3]:
            raise ValueError(f"{sample_id}: prior estimation requires all four cyclic permutations")
        probabilities = [[float(row[f"prob_{label}"]) for label in OPTION_LABELS] for row in ordered]
        prior = estimate_sample_prior(probabilities, epsilon=epsilon)
        sample_priors.append(prior.tolist())
        sample_prior_rows.append(
            {
                "sample_id": sample_id,
                **{f"prior_{label}": float(prior[index]) for index, label in enumerate(OPTION_LABELS)},
            }
        )
    global_prior = estimate_global_prior(sample_priors).tolist()
    return sample_prior_rows, global_prior

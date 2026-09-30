from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _distribution(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (4,):
        raise ValueError(f"{name} must contain four values")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.isclose(array.sum(), 1.0, atol=1e-6):
        raise ValueError(f"{name} must sum to one")
    return array


def apply_pride_correction(
    observed_probabilities: Sequence[float],
    prior: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Implements Zheng et al. (2024), Equation 8."""
    observed = _distribution(observed_probabilities, "observed_probabilities")
    prior_array = _distribution(prior, "prior")
    corrected = observed / np.clip(prior_array, epsilon, None)
    total = corrected.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PriDe correction produced an invalid normalization constant")
    return corrected / total


def cyclic_content_average(
    permutation_probabilities: Sequence[Sequence[float]],
    display_to_content: Sequence[Sequence[int]],
) -> np.ndarray:
    probabilities = np.asarray(permutation_probabilities, dtype=np.float64)
    mappings = np.asarray(display_to_content, dtype=np.int64)
    if probabilities.shape != (4, 4) or mappings.shape != (4, 4):
        raise ValueError("Cyclic debiasing requires 4 x 4 probabilities and mappings")
    if any(set(row.tolist()) != set(range(4)) for row in mappings):
        raise ValueError("Each permutation mapping must contain content indices 0..3 exactly once")
    content_probabilities = np.zeros((4, 4), dtype=np.float64)
    for permutation_index in range(4):
        for display_index, content_index in enumerate(mappings[permutation_index]):
            content_probabilities[permutation_index, content_index] = probabilities[
                permutation_index, display_index
            ]
    averaged = content_probabilities.mean(axis=0)
    return averaged / averaged.sum()


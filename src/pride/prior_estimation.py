from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

import numpy as np


def _normalize_probability_matrix(probabilities: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Expected an N x 4 probability matrix, got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to one")
    return array


def estimate_sample_prior(
    permutation_probabilities: Sequence[Sequence[float]],
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Implements Zheng et al. (2024), Equation 7, for one sample."""
    probabilities = _normalize_probability_matrix(permutation_probabilities)
    if probabilities.shape[0] != 4:
        raise ValueError("Standard PriDe estimation requires all four cyclic permutations")
    mean_log_probabilities = np.log(np.clip(probabilities, epsilon, 1.0)).mean(axis=0)
    shifted = mean_log_probabilities - mean_log_probabilities.max()
    prior = np.exp(shifted)
    return prior / prior.sum()


def estimate_global_prior(sample_priors: Sequence[Sequence[float]]) -> np.ndarray:
    priors = _normalize_probability_matrix(sample_priors)
    if priors.shape[0] == 0:
        raise ValueError("At least one sample prior is required")
    global_prior = priors.mean(axis=0)
    return global_prior / global_prior.sum()


def estimation_sample_count(
    dataset_size: int,
    alpha: float,
    *,
    rounding: str = "floor",
    minimum: int = 1,
) -> int:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    raw = dataset_size * alpha
    functions = {"ceil": math.ceil, "floor": math.floor, "round": round}
    try:
        count = int(functions[rounding](raw))
    except KeyError as exc:
        raise ValueError("rounding must be one of: ceil, floor, round") from exc
    return min(dataset_size, max(minimum, count))


def select_estimation_ids(
    sample_ids: Iterable[str],
    alpha: float,
    seed: int,
    *,
    rounding: str = "floor",
    minimum: int = 1,
    sort_before_sampling: bool = True,
) -> list[str]:
    ids = list(sample_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    population = sorted(ids) if sort_before_sampling else ids
    count = estimation_sample_count(len(population), alpha, rounding=rounding, minimum=minimum)
    selected = random.Random(seed).sample(population, count)
    return sorted(selected)

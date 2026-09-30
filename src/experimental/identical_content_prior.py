from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def identical_content_options(content: str) -> tuple[str, str, str, str]:
    """Construct the diagnostic option list; this is not a PriDe permutation."""
    value = content.strip()
    if not value:
        raise ValueError("Diagnostic content must not be empty")
    return (value, value, value, value)


def estimate_identical_content_prior(probability_rows: Sequence[Sequence[float]]) -> np.ndarray:
    probabilities = np.asarray(probability_rows, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4 or probabilities.shape[0] == 0:
        raise ValueError("Expected one or more four-option probability rows")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must be a valid distribution")
    prior = probabilities.mean(axis=0)
    return prior / prior.sum()


def compare_priors(
    standard_pride_prior: Sequence[float],
    identical_content_prior: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    standard = np.asarray(standard_pride_prior, dtype=np.float64)
    experimental = np.asarray(identical_content_prior, dtype=np.float64)
    if standard.shape != (4,) or experimental.shape != (4,):
        raise ValueError("Both priors must contain four values")
    safe_standard = np.clip(standard, epsilon, None)
    safe_experimental = np.clip(experimental, epsilon, None)
    denominator = np.linalg.norm(standard) * np.linalg.norm(experimental)
    correlation = float(np.corrcoef(standard, experimental)[0, 1])
    return {
        "l1_distance": float(np.abs(standard - experimental).sum()),
        "kl_standard_to_identical": float(
            np.sum(safe_standard * np.log(safe_standard / safe_experimental))
        ),
        "cosine_similarity": float(np.dot(standard, experimental) / denominator),
        "pearson_correlation": correlation,
    }


def comparison_report(
    standard_pride_prior: Sequence[float],
    identical_prior: Sequence[float],
    *,
    standard_downstream: dict[str, float],
    identical_downstream: dict[str, float],
    epsilon: float = 1e-12,
) -> dict[str, float]:
    required = {"accuracy", "rstd"}
    if not required.issubset(standard_downstream) or not required.issubset(identical_downstream):
        raise ValueError("Both downstream metric mappings must contain accuracy and rstd")
    report = compare_priors(
        standard_pride_prior, identical_prior, epsilon=epsilon
    )
    report.update(
        standard_pride_accuracy=float(standard_downstream["accuracy"]),
        standard_pride_rstd=float(standard_downstream["rstd"]),
        identical_prior_accuracy=float(identical_downstream["accuracy"]),
        identical_prior_rstd=float(identical_downstream["rstd"]),
    )
    return report

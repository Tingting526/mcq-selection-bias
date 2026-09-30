import numpy as np

from src.data.base import MCQSample
from src.evaluation.pride_eval import estimate_prior_from_permutation_records
from src.permutations.cyclic import cyclic_permutations
from src.pride.prior_estimation import (
    estimate_global_prior,
    estimate_sample_prior,
    select_estimation_ids,
)


def synthetic_observed_probabilities() -> tuple[np.ndarray, np.ndarray]:
    sample = MCQSample("q", "test", "Question", ("o0", "o1", "o2", "o3"), 0)
    content_belief = np.asarray([0.55, 0.25, 0.15, 0.05])
    prior = np.asarray([0.4, 0.3, 0.2, 0.1])
    observed = []
    for permutation in cyclic_permutations(sample):
        values = np.asarray(
            [prior[d] * content_belief[c] for d, c in enumerate(permutation.display_to_content)]
        )
        observed.append(values / values.sum())
    return np.asarray(observed), prior


def test_sample_prior_recovers_multiplicative_option_bias() -> None:
    observed, expected_prior = synthetic_observed_probabilities()
    actual = estimate_sample_prior(observed)
    assert np.allclose(actual, expected_prior)
    assert np.isclose(actual.sum(), 1.0)


def test_global_prior_is_arithmetic_mean_of_sample_priors() -> None:
    actual = estimate_global_prior([[0.4, 0.3, 0.2, 0.1], [0.2, 0.2, 0.2, 0.4]])
    assert np.allclose(actual, [0.3, 0.25, 0.2, 0.25])


def test_prior_record_estimator_does_not_require_gold_labels() -> None:
    observed, expected_prior = synthetic_observed_probabilities()
    rows = []
    for permutation_id, probabilities in enumerate(observed):
        rows.append(
            {
                "sample_id": "q",
                "permutation_id": permutation_id,
                **{f"prob_{label}": probabilities[index] for index, label in enumerate("ABCD")},
            }
        )
    _, global_prior = estimate_prior_from_permutation_records(rows)
    assert np.allclose(global_prior, expected_prior)


def test_estimation_sampling_is_deterministic_and_order_independent() -> None:
    ids = [f"q{i}" for i in range(20)]
    first = select_estimation_ids(ids, 0.2, 42)
    second = select_estimation_ids(reversed(ids), 0.2, 42)
    assert first == second
    assert len(first) == 4


def test_default_estimation_rounding_is_floor() -> None:
    ids = [f"q{i}" for i in range(21)]
    assert len(select_estimation_ids(ids, 0.05, 42)) == 1

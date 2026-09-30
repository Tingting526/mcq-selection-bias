import numpy as np

from src.experimental.identical_content_prior import (
    comparison_report,
    estimate_identical_content_prior,
    identical_content_options,
)


def test_identical_content_estimator_is_separate_and_normalized() -> None:
    assert identical_content_options(" X ") == ("X", "X", "X", "X")
    prior = estimate_identical_content_prior(
        [[0.4, 0.3, 0.2, 0.1], [0.2, 0.3, 0.3, 0.2]]
    )
    assert np.allclose(prior, [0.3, 0.3, 0.25, 0.15])
    assert np.isclose(prior.sum(), 1.0)


def test_comparison_report_includes_prior_and_downstream_metrics() -> None:
    report = comparison_report(
        [0.4, 0.3, 0.2, 0.1],
        [0.35, 0.3, 0.2, 0.15],
        standard_downstream={"accuracy": 0.7, "rstd": 0.04},
        identical_downstream={"accuracy": 0.68, "rstd": 0.05},
    )
    assert report["l1_distance"] > 0
    assert report["standard_pride_accuracy"] == 0.7
    assert report["identical_prior_rstd"] == 0.05


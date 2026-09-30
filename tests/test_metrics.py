import numpy as np

from src.metrics.selection_bias import permutation_accuracy_metrics, selection_bias_metrics


def test_recall_rstd_and_prediction_frequencies() -> None:
    rows = [
        {"gold_label": "A", "predicted_label": "A"},
        {"gold_label": "B", "predicted_label": "A"},
        {"gold_label": "C", "predicted_label": "C"},
        {"gold_label": "D", "predicted_label": "D"},
    ]
    metrics = selection_bias_metrics(rows)
    assert metrics["accuracy"] == 0.75
    assert [metrics[f"recall_{label}"] for label in "ABCD"] == [1.0, 0.0, 1.0, 1.0]
    assert np.isclose(metrics["rstd"], np.std([1.0, 0.0, 1.0, 1.0], ddof=0))
    assert metrics["prediction_count_A"] == 2
    assert metrics["prediction_frequency_A"] == 0.5


def test_permutation_accuracy_variation() -> None:
    rows = []
    patterns = ([True, True], [True, False], [False, False], [True, True])
    for permutation_id, values in enumerate(patterns):
        rows.extend(
            {"permutation_id": permutation_id, "correct": value}
            for value in values
        )
    metrics = permutation_accuracy_metrics(rows)
    expected = np.asarray([1.0, 0.5, 0.0, 1.0])
    assert np.isclose(metrics["accuracy_permutation_variance"], np.var(expected, ddof=0))
    assert metrics["accuracy_permutation_min"] == 0.0
    assert metrics["accuracy_permutation_max"] == 1.0
    assert metrics["accuracy_permutation_range"] == 1.0


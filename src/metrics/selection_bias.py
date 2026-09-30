from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src import OPTION_LABELS
from .accuracy import accuracy


def selection_bias_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    gold_key: str = "gold_label",
    prediction_key: str = "predicted_label",
) -> dict[str, float | int]:
    gold = [str(row[gold_key]) for row in rows]
    predicted = [str(row[prediction_key]) for row in rows]
    result: dict[str, float | int] = {"n": len(rows), "accuracy": accuracy(gold, predicted)}
    recalls: list[float] = []
    for label in OPTION_LABELS:
        indices = [index for index, value in enumerate(gold) if value == label]
        recall = (
            sum(predicted[index] == label for index in indices) / len(indices)
            if indices
            else float("nan")
        )
        result[f"recall_{label}"] = recall
        recalls.append(recall)
    result["rstd"] = float(np.std(recalls, ddof=0)) if not any(np.isnan(recalls)) else float("nan")
    frequencies = Counter(predicted)
    for label in OPTION_LABELS:
        result[f"prediction_count_{label}"] = frequencies[label]
        result[f"prediction_frequency_{label}"] = frequencies[label] / len(rows) if rows else float("nan")
    return result


def permutation_accuracy_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    permutation_key: str = "permutation_id",
    correct_key: str = "correct",
) -> dict[str, float]:
    accuracies: list[float] = []
    result: dict[str, float] = {}
    for permutation_id in range(4):
        subset = [row for row in rows if int(row[permutation_key]) == permutation_id]
        value = (
            sum(bool(row[correct_key]) for row in subset) / len(subset)
            if subset
            else float("nan")
        )
        result[f"accuracy_permutation_{permutation_id}"] = value
        accuracies.append(value)
    if any(np.isnan(accuracies)):
        result.update(
            accuracy_permutation_variance=float("nan"),
            accuracy_permutation_std=float("nan"),
            accuracy_permutation_min=float("nan"),
            accuracy_permutation_max=float("nan"),
            accuracy_permutation_range=float("nan"),
        )
        return result
    array = np.asarray(accuracies, dtype=np.float64)
    result.update(
        accuracy_permutation_variance=float(np.var(array, ddof=0)),
        accuracy_permutation_std=float(np.std(array, ddof=0)),
        accuracy_permutation_min=float(array.min()),
        accuracy_permutation_max=float(array.max()),
        accuracy_permutation_range=float(array.max() - array.min()),
    )
    return result


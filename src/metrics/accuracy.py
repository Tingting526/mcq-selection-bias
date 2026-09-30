from __future__ import annotations

from collections.abc import Sequence


def accuracy(gold: Sequence[object], predicted: Sequence[object]) -> float:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have equal length")
    if not gold:
        return float("nan")
    return sum(expected == actual for expected, actual in zip(gold, predicted, strict=True)) / len(gold)


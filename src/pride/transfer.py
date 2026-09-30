from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def transfer_summary(
    *,
    source_dataset: str,
    target_dataset: str,
    model: str,
    alpha: float,
    prior: list[float],
    baseline: Mapping[str, Any],
    pride: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "transfer_type": "within_dataset" if source_dataset == target_dataset else "cross_dataset",
        "model": model,
        "alpha": alpha,
        **{f"prior_{label}": prior[index] for index, label in enumerate("ABCD")},
        "target_baseline_accuracy": baseline["accuracy"],
        "target_baseline_rstd": baseline["rstd"],
        "target_pride_accuracy": pride["accuracy"],
        "target_pride_rstd": pride["rstd"],
        "delta_accuracy": pride["accuracy"] - baseline["accuracy"],
        "delta_rstd": pride["rstd"] - baseline["rstd"],
    }


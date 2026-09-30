from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

from .base import MCQSample
from .medmcqa import normalize_medmcqa
from .medqa import normalize_medqa
from .mmlu import normalize_mmlu
from .preprocessing import (
    PreprocessingPolicy,
    load_preprocessed_artifact,
    preprocess_rows,
    sample_set_hash,
)


ADAPTERS: dict[str, Callable[[Mapping[str, Any], int], MCQSample]] = {
    "medqa": normalize_medqa,
    "medmcqa": normalize_medmcqa,
    "mmlu": normalize_mmlu,
}


def normalize_row(adapter: str, row: Mapping[str, Any], row_index: int) -> MCQSample:
    try:
        normalizer = ADAPTERS[adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset adapter: {adapter}") from exc
    return normalizer(row, row_index)


def load_samples(
    dataset_config: Mapping[str, Any],
    split: str,
    *,
    limit: int | None = None,
    start_index: int = 0,
    preprocessed: str | Path | None = None,
    preprocessing_config: Mapping[str, Any] | None = None,
) -> tuple[list[MCQSample], str | None]:
    samples, fingerprint, _ = load_samples_with_metadata(
        dataset_config,
        split,
        limit=limit,
        start_index=start_index,
        preprocessed=preprocessed,
        preprocessing_config=preprocessing_config,
    )
    return samples, fingerprint


def load_samples_with_metadata(
    dataset_config: Mapping[str, Any],
    split: str,
    *,
    limit: int | None = None,
    start_index: int = 0,
    preprocessed: str | Path | None = None,
    preprocessing_config: Mapping[str, Any] | None = None,
) -> tuple[list[MCQSample], str | None, dict[str, Any]]:
    """Load the audited experiment population, selecting only after filtering.

    Downstream evaluation uses this entry point so baseline, permutation, and
    PriDe runs cannot accidentally operate on different preprocessing rules.
    """
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    if preprocessed is not None:
        samples, manifest = load_preprocessed_artifact(preprocessed)
        expected_alias = str(dataset_config.get("alias") or dataset_config["adapter"])
        if manifest.get("dataset_alias") != expected_alias:
            raise ValueError(
                f"Preprocessed artifact dataset is {manifest.get('dataset_alias')!r}, "
                f"expected {expected_alias!r}"
            )
        if manifest.get("dataset_split") != split:
            raise ValueError(
                f"Preprocessed artifact split is {manifest.get('dataset_split')!r}, "
                f"expected {split!r}"
            )
        stop = None if limit is None else start_index + limit
        selected = samples[start_index:stop]
        metadata = dict(manifest)
        metadata.update(
            preprocessing_source="artifact",
            preprocessing_artifact=str(Path(preprocessed).resolve()),
            downstream_selection_start_index=start_index,
            downstream_selection_limit=limit,
            downstream_selected_samples=len(selected),
            analysis_sample_set_sha256=sample_set_hash(selected),
        )
        return selected, manifest.get("dataset_fingerprint"), metadata

    from datasets import load_dataset
    from src.utils.config import load_yaml

    dataset = load_dataset(
        dataset_config["repo"],
        dataset_config.get("config"),
        split=split,
        revision=dataset_config.get("revision"),
    )
    if preprocessing_config is None:
        preprocessing_config = load_yaml("configs/preprocessing/thesis.yaml")
    result = preprocess_rows(
        dataset,
        adapter=str(dataset_config["adapter"]),
        normalizer=normalize_row,
        policy=PreprocessingPolicy.from_mapping(preprocessing_config),
    ).select(start_index, limit)
    metadata = {
        **result.report,
        "preprocessing_source": "inline",
        "preprocessing_artifact": None,
        "analysis_sample_set_sha256": result.report["selected_sample_set_sha256"],
    }
    return result.samples, getattr(dataset, "_fingerprint", None), metadata

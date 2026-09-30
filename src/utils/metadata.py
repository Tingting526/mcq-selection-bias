from __future__ import annotations

import importlib.metadata
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping

from .config import PROJECT_ROOT


TRACKED_PACKAGES = ("accelerate", "datasets", "numpy", "pandas", "pyarrow", "torch", "transformers")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_metadata(
    *,
    model_config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    split: str,
    seed: int,
    batch_size: int,
    prompt_template_version: str,
    dataset_fingerprint: str | None,
    resolved_model_revision: str | None,
    resolved_tokenizer_revision: str | None,
    resolved_device: Any = None,
    start_index: int = 0,
    limit: int | None = None,
    alpha: float | None = None,
    estimation_ids: list[str] | None = None,
    scorer_backend: str = "real_transformers",
    model_class: str | None = None,
    tokenizer_class: str | None = None,
    option_token_ids: list[int] | None = None,
    preprocessing_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "scorer_backend": scorer_backend,
        "model_repo": model_config["model_repo"],
        "model_class": model_class or model_config.get("model_class", "AutoModelForCausalLM"),
        "tokenizer_class": tokenizer_class,
        "tokenizer_backend": model_config.get("tokenizer_backend", "auto"),
        "chat_template_kwargs": dict(model_config.get("chat_template_kwargs") or {}),
        "option_token_ids": option_token_ids,
        "model_revision_requested": model_config.get("model_revision"),
        "model_revision_resolved": resolved_model_revision,
        "tokenizer_repo": model_config.get("tokenizer_repo", model_config["model_repo"]),
        "tokenizer_revision_requested": model_config.get("tokenizer_revision"),
        "tokenizer_revision_resolved": resolved_tokenizer_revision,
        "dataset_repo": dataset_config["repo"],
        "dataset_revision": dataset_config.get("revision"),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_split": split,
        "selection_start_index": start_index,
        "selection_limit": limit,
        "seed": seed,
        "dtype": model_config.get("dtype"),
        "device_requested": model_config.get("device_map"),
        "device_resolved": resolved_device,
        "batch_size": batch_size,
        "prompt_template_version": prompt_template_version,
        "estimation_fraction_alpha": alpha,
        "prior_estimation_sample_ids": estimation_ids or [],
        "preprocessing": dict(preprocessing_metadata or {}),
        "package_versions": package_versions(),
    }

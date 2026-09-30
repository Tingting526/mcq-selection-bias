from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT
from src.data.preprocessing import load_preprocessed_artifact
from src.utils.config import load_yaml, named_config


DEFAULT_DATASETS = ("medqa", "medmcqa", "mmlu")
PREPROCESSING_TESTS = (
    "tests/test_dataset_normalization.py",
    "tests/test_permutations.py",
    "tests/test_preprocessing.py",
)


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    dataset: str
    split: str
    raw_rows: int
    accepted_rows: int
    excluded_rows: int
    prompt_rows: int
    artifact: Path


def _resolve_output_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _unique_artifact_path(
    output_root: Path,
    dataset: str,
    split: str,
    preprocessing_version: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent = output_root / dataset
    parent.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp}_{split}_{preprocessing_version}"
    candidate = parent / stem
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{stem}_{suffix:02d}"
        suffix += 1
    return candidate


def _run_one(
    dataset: str,
    *,
    output_root: Path,
    preprocessing_path: str,
    experiment_path: str,
    start_index: int,
    limit: int | None,
) -> ValidationSummary:
    dataset_config = named_config("datasets", dataset)
    split = str(dataset_config.get("default_split") or "")
    if not split:
        raise ValueError(f"{dataset}: no default split is configured")
    preprocessing = load_yaml(preprocessing_path)
    output = _unique_artifact_path(
        output_root,
        dataset,
        split,
        str(preprocessing["version"]),
    )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "preprocess_data.py"),
        "--dataset",
        dataset,
        "--preprocessing",
        preprocessing_path,
        "--experiment",
        experiment_path,
        "--start-index",
        str(start_index),
        "--output",
        str(output),
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))

    print(f"\n=== {dataset.upper()} ({split}) ===", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    samples, manifest = load_preprocessed_artifact(output, verify_hashes=True)
    prompts = pd.read_parquet(output / "cyclic_prompts.parquet")
    excluded = pd.read_parquet(output / "excluded_rows.parquet")
    if len(samples) != int(manifest["selected_samples"]):
        raise ValueError(f"{dataset}: samples.parquet count does not match manifest")
    if len(excluded) != int(manifest["excluded_rows"]):
        raise ValueError(f"{dataset}: excluded_rows.parquet count does not match manifest")
    if len(prompts) != 4 * len(samples):
        raise ValueError(f"{dataset}: expected exactly four prompts per sample")
    if samples and not prompts.groupby("sample_id").size().eq(4).all():
        raise ValueError(f"{dataset}: at least one sample does not have four prompt rows")
    if samples:
        permutation_ids = prompts.groupby("sample_id")["permutation_id"].apply(
            lambda values: set(values.astype(int))
        )
        if not permutation_ids.map(lambda values: values == {0, 1, 2, 3}).all():
            raise ValueError(f"{dataset}: invalid cyclic permutation IDs")

    print("Validation: checksums, counts, IDs and four permutations per sample are OK.")
    return ValidationSummary(
        dataset=dataset,
        split=split,
        raw_rows=int(manifest["raw_rows"]),
        accepted_rows=len(samples),
        excluded_rows=len(excluded),
        prompt_rows=len(prompts),
        artifact=output,
    )


def _print_summary(rows: list[ValidationSummary]) -> None:
    print("\n=== COMPLETE PREPROCESSING SUMMARY ===")
    print(
        f"{'dataset':10s} {'split':12s} {'raw':>8s} "
        f"{'accepted':>10s} {'excluded':>10s} {'prompts':>10s}"
    )
    for row in rows:
        print(
            f"{row.dataset:10s} {row.split:12s} {row.raw_rows:8,d} "
            f"{row.accepted_rows:10,d} {row.excluded_rows:10,d} "
            f"{row.prompt_rows:10,d}"
        )
    print("\nArtifacts:")
    for row in rows:
        print(f"- {row.dataset}: {row.artifact}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run and validate the complete model-independent preprocessing for "
            "MedQA, MedMCQA, and MMLU"
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        help="Accepted samples per dataset; omit for the complete splits",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/preprocessed"),
    )
    parser.add_argument(
        "--preprocessing",
        default="configs/preprocessing/thesis.yaml",
    )
    parser.add_argument(
        "--experiment",
        default="configs/experiments/thesis.yaml",
    )
    parser.add_argument("--no-tests", action="store_true")
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Run the full CPU test suite instead of only preprocessing tests",
    )
    args = parser.parse_args()
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    output_root = _resolve_output_root(args.output_root)
    summaries = [
        _run_one(
            dataset,
            output_root=output_root,
            preprocessing_path=args.preprocessing,
            experiment_path=args.experiment,
            start_index=args.start_index,
            limit=args.limit,
        )
        for dataset in args.datasets
    ]
    _print_summary(summaries)

    if not args.no_tests:
        test_targets = [] if args.all_tests else list(PREPROCESSING_TESTS)
        print("\n=== TESTS ===", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", *test_targets],
            cwd=PROJECT_ROOT,
            check=True,
        )
        print("All requested tests passed.")


if __name__ == "__main__":
    main()

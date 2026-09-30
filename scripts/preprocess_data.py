from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.data.preprocessing import (
    PreprocessingPolicy,
    preprocess_rows,
    write_preprocessed_artifact,
)
from src.data.registry import normalize_row
from src.prompts.mcq import MCQPrompt, PromptConfig
from src.utils.config import load_yaml, named_config


def _resolved_dataset_revision(repo: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import HfApi

        return str(HfApi().dataset_info(repo, revision=revision).sha)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the complete audited dataset/prompt artifact before model evaluation"
    )
    parser.add_argument("--dataset", required=True, help="Alias from configs/datasets")
    parser.add_argument("--split")
    parser.add_argument("--preprocessing", default="configs/preprocessing/thesis.yaml")
    parser.add_argument("--experiment", default="configs/experiments/thesis.yaml")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    from datasets import load_dataset

    dataset_config = named_config("datasets", args.dataset)
    preprocessing_config = load_yaml(args.preprocessing)
    experiment = load_yaml(args.experiment)
    split = str(args.split or dataset_config.get("default_split") or "")
    if not split:
        raise ValueError("A split must be supplied by --split or the dataset configuration")

    dataset = load_dataset(
        dataset_config["repo"],
        dataset_config.get("config"),
        split=split,
        revision=dataset_config.get("revision"),
    )
    policy = PreprocessingPolicy.from_mapping(preprocessing_config)
    full_result = preprocess_rows(
        dataset,
        adapter=str(dataset_config["adapter"]),
        normalizer=normalize_row,
        policy=policy,
    )
    result = full_result.select(args.start_index, args.limit)
    prompt_builder = MCQPrompt(PromptConfig.from_mapping(experiment["prompt"]))

    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            PROJECT_ROOT
            / "outputs"
            / "preprocessed"
            / args.dataset
            / f"{timestamp}_{split}_{policy.version}"
        )
    else:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    artifact = write_preprocessed_artifact(
        result,
        output,
        prompt_builder=prompt_builder,
        dataset_metadata={
            "dataset_alias": args.dataset,
            "dataset_repo": dataset_config["repo"],
            "dataset_config": dataset_config.get("config"),
            "dataset_revision_requested": dataset_config.get("revision"),
            "dataset_revision_resolved": _resolved_dataset_revision(
                str(dataset_config["repo"]), dataset_config.get("revision")
            ),
            "dataset_split": split,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        },
    )

    print(f"Raw rows: {result.report['raw_rows']:,}")
    print(f"Accepted before selection: {result.report['accepted_rows_before_selection']:,}")
    print(f"Excluded: {result.report['excluded_rows']:,}")
    print(f"Selected artifact rows: {result.report['selected_samples']:,}")
    print(f"Exclusions by reason: {result.report['excluded_by_reason']}")
    print(f"Preprocessed artifact: {artifact}")


if __name__ == "__main__":
    main()

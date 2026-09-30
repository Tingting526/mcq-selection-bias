from __future__ import annotations

import argparse
import json

from _bootstrap import PROJECT_ROOT  # noqa: F401
from src.data.registry import normalize_row
from src.utils.config import named_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache one configured dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split")
    args = parser.parse_args()

    from datasets import get_dataset_split_names, load_dataset

    config = named_config("datasets", args.dataset)
    repo = config["repo"]
    hf_config = config.get("config")
    revision = config.get("revision")
    split = str(args.split or config.get("default_split") or "")
    if not split:
        raise ValueError("A split must be provided either by --split or dataset default_split")

    print(f"repository        : {repo}")
    print(f"config            : {hf_config}")
    print(f"revision          : {revision}")

    try:
        splits = get_dataset_split_names(repo, hf_config, revision=revision)
    except Exception as exc:  # pragma: no cover - network/version dependent
        splits = f"(could not enumerate: {exc.__class__.__name__})"
    print(f"available splits  : {splits}")
    print(f"selected split    : {split}")

    dataset = load_dataset(repo, hf_config, split=split, revision=revision)
    print(f"number of samples : {len(dataset):,}")
    print(f"dataset fingerprint: {getattr(dataset, '_fingerprint', None)}")

    raw_example = dataset[0]
    print("\n--- one RAW example (as delivered by Hugging Face) ---")
    print(json.dumps(raw_example, indent=2, ensure_ascii=False)[:1200])

    normalized = normalize_row(config["adapter"], raw_example, 0)
    print("\n--- one NORMALIZED MCQSample example ---")
    print(json.dumps(normalized.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

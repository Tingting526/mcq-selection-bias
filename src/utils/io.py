from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import PROJECT_ROOT


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, default=_json_default)
        stream.write("\n")
    temporary.replace(output)
    return output


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def unique_path(directory: str | Path, stem: str, suffix: str) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = root / f"{stem}_{counter:02d}{suffix}"
        counter += 1
    return candidate


class RunStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.parts_dir = self.run_dir / "parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        kind: str,
        model_alias: str,
        dataset_alias: str,
        *,
        resume: str | Path | None = None,
    ) -> "RunStore":
        if resume is not None:
            run_dir = Path(resume)
            if not run_dir.exists():
                raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
            return cls(run_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = PROJECT_ROOT / "outputs" / "raw" / kind
        run_dir = unique_path(base, f"{timestamp}_{model_alias}_{dataset_alias}", "")
        run_dir.mkdir(parents=True, exist_ok=False)
        return cls(run_dir)

    def part_files(self) -> list[Path]:
        return sorted(self.parts_dir.glob("part-*.parquet"))

    def write_records(self, records: list[dict[str, Any]]) -> Path | None:
        if not records:
            return None
        path = self.parts_dir / f"part-{len(self.part_files()):06d}.parquet"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite result partition: {path}")
        temporary = path.with_suffix(".parquet.tmp")
        pd.DataFrame.from_records(records).to_parquet(temporary, index=False)
        temporary.replace(path)
        return path

    def read_frame(self) -> pd.DataFrame:
        files = self.part_files()
        if not files:
            return pd.DataFrame()
        return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    def read_records(self) -> list[dict[str, Any]]:
        return self.read_frame().to_dict(orient="records")

    def completed_ids(self) -> set[str]:
        frame = self.read_frame()
        return set(frame["sample_id"].astype(str)) if "sample_id" in frame else set()

    def completed_permutations(self) -> set[tuple[str, int]]:
        frame = self.read_frame()
        if not {"sample_id", "permutation_id"}.issubset(frame.columns):
            return set()
        return set(zip(frame["sample_id"].astype(str), frame["permutation_id"].astype(int), strict=True))

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        path = self.run_dir / "metadata.json"
        if path.exists():
            existing = read_json(path)
            keys = (
                "scorer_backend",
                "model_revision_resolved",
                "tokenizer_revision_resolved",
                "option_token_ids",
                "model_repo",
                "model_revision_requested",
                "tokenizer_repo",
                "dataset_repo",
                "dataset_revision",
                "dataset_split",
                "selection_start_index",
                "selection_limit",
                "seed",
                "prompt_template_version",
                "chat_template_kwargs",
                "estimation_fraction_alpha",
                "prior_estimation_sample_ids",
                "prior_source_dataset",
                "prior_global",
                "baseline_source_sha256",
                "baseline_source_run",
                "preprocessing",
                "target_dataset_repo",
                "target_dataset_revision",
                "target_dataset_split",
                "target_selection_limit",
                "target_preprocessing",
            )
            mismatches = [key for key in keys if existing.get(key) != metadata.get(key)]
            if mismatches:
                raise ValueError(
                    "Resume metadata does not match this invocation for: " + ", ".join(mismatches)
                )
            return path
        return write_json(path, metadata)

    def save_debug_prompts(self, prompts: Iterable[dict[str, Any]]) -> Path:
        path = self.run_dir / "debug_prompts.json"
        if path.exists():
            return path
        return write_json(path, list(prompts))

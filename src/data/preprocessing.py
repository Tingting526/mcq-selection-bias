from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from src.data.base import MCQSample
from src.permutations.cyclic import cyclic_permutations
from src.prompts.mcq import MCQPrompt
from src.utils.io import read_json, write_json


PREPROCESSED_SCHEMA_VERSION = "mcq-preprocessed-v1"

_RELATIVE_POSITION_REFERENCE = re.compile(
    r"\b(?:all|none|both|either|neither)\s+of\s+(?:the\s+)?(?:above|below)\b",
    flags=re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z]+")
_LABEL_COMBINATION_WORDS = {
    "all",
    "and",
    "are",
    "both",
    "correct",
    "either",
    "false",
    "incorrect",
    "is",
    "neither",
    "nether",  # common typo present in some MCQ corpora
    "nor",
    "of",
    "only",
    "option",
    "options",
    "or",
    "the",
    "true",
}


@dataclass(frozen=True, slots=True)
class PreprocessingPolicy:
    version: str = "mcq-preprocessing-v1"
    exclude_relative_position_references: bool = True
    exclude_option_label_combinations: bool = True
    exclude_duplicate_option_texts: bool = True
    normalization_errors: str = "quarantine"
    duplicate_sample_ids: str = "quarantine_later"
    exact_duplicate_samples: str = "keep_and_report"
    selection_stage: str = "after_filtering"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PreprocessingPolicy":
        if value is None:
            return cls()
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(
                "Unknown preprocessing configuration keys: " + ", ".join(unknown)
            )
        defaults = cls()
        policy = cls(
            **{field: value.get(field, getattr(defaults, field)) for field in known}
        )
        if not policy.version.strip():
            raise ValueError("Preprocessing version must not be empty")
        if policy.normalization_errors not in {"quarantine", "error"}:
            raise ValueError("normalization_errors must be 'quarantine' or 'error'")
        if policy.duplicate_sample_ids not in {"quarantine_later", "error"}:
            raise ValueError("duplicate_sample_ids must be 'quarantine_later' or 'error'")
        if policy.exact_duplicate_samples != "keep_and_report":
            raise ValueError("exact_duplicate_samples currently supports only 'keep_and_report'")
        if policy.selection_stage != "after_filtering":
            raise ValueError("selection_stage must be 'after_filtering'")
        return policy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedSample:
    source_index: int
    sample: MCQSample
    content_hash: str


@dataclass(frozen=True, slots=True)
class ExcludedRow:
    source_index: int
    sample_id: str | None
    reason_codes: tuple[str, ...]
    detail: str | None
    question: str | None
    options: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "sample_id": self.sample_id,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "question": self.question,
            "options": list(self.options),
        }


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    prepared: tuple[PreparedSample, ...]
    excluded: tuple[ExcludedRow, ...]
    report: dict[str, Any]

    @property
    def samples(self) -> list[MCQSample]:
        return [item.sample for item in self.prepared]

    def select(self, start_index: int = 0, limit: int | None = None) -> "PreprocessingResult":
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        stop = None if limit is None else start_index + limit
        selected = self.prepared[start_index:stop]
        report = dict(self.report)
        report.update(
            selection_start_index=start_index,
            selection_limit=limit,
            selected_samples=len(selected),
            selected_sample_set_sha256=_sample_set_hash(selected),
        )
        return PreprocessingResult(selected, self.excluded, report)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_hash(sample: MCQSample) -> str:
    return _sha256(_canonical_json({"question": sample.question, "options": sample.options}))


def _sample_set_hash(prepared: Sequence[PreparedSample]) -> str:
    payload = [
        {
            "sample_id": item.sample.sample_id,
            "content_hash": item.content_hash,
            "gold_index": item.sample.gold_index,
        }
        for item in prepared
    ]
    return _sha256(_canonical_json(payload))


def sample_set_hash(samples: Sequence[MCQSample]) -> str:
    """Stable identity of the exact ordered population used by an experiment."""
    prepared = tuple(
        PreparedSample(index, sample, _content_hash(sample))
        for index, sample in enumerate(samples)
    )
    return _sample_set_hash(prepared)


def has_relative_position_reference(option: str) -> bool:
    """Whether an option refers to its current relative position (above/below)."""
    return _RELATIVE_POSITION_REFERENCE.search(option) is not None


def has_option_label_combination(option: str) -> bool:
    """Detect answer choices whose meaning is a combination of displayed A-D IDs.

    The PriDe reference code uses substring checks such as ``"A and B"``. Here
    we additionally require the whole choice to consist of combination language,
    avoiding false positives such as ``"Vitamin A and D"`` or propositions in a
    formal-logic explanation.
    """
    words = _WORD.findall(option)
    labels = [word for word in words if len(word) == 1 and word in "ABCD"]
    if len(set(labels)) < 2:
        return False
    non_labels = [
        word.casefold()
        for word in words
        if not (len(word) == 1 and word in "ABCD")
    ]
    return all(word in _LABEL_COMBINATION_WORDS for word in non_labels)


def has_display_label_reference(options: Sequence[str]) -> bool:
    """Whether combination choices most likely refer to displayed option IDs.

    If three or four choices are themselves bare letter combinations or
    sequences, the letters are part of the problem's answer vocabulary (for
    example an ordering task or Hb-chain combination), not a meta-choice that
    points to other displayed answers. A minority of combination choices is the
    unsafe pattern targeted by PriDe's preprocessing.
    """
    count = sum(has_option_label_combination(option) for option in options)
    return 0 < count < 3


def duplicate_option_texts(options: Sequence[str]) -> bool:
    normalized = [" ".join(option.split()).casefold() for option in options]
    return len(set(normalized)) != len(normalized)


def _raw_preview(
    row: Mapping[str, Any],
) -> tuple[str | None, str | None, tuple[str, ...]]:
    raw_id = row.get("id")
    raw_question = row.get("question", row.get("sent1"))
    if (
        "choices" in row
        and isinstance(row.get("choices"), Sequence)
        and not isinstance(row.get("choices"), (str, bytes))
    ):
        options = tuple(str(value) for value in row["choices"])
    else:
        keys = (
            ("opa", "opb", "opc", "opd")
            if "opa" in row
            else tuple(f"ending{i}" for i in range(4))
        )
        options = tuple(
            str(row[key]) for key in keys if key in row and row[key] is not None
        )
    return (
        None if raw_id is None else str(raw_id),
        None if raw_question is None else str(raw_question),
        options,
    )


def preprocess_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    adapter: str,
    normalizer: Callable[[str, Mapping[str, Any], int], MCQSample],
    policy: PreprocessingPolicy | None = None,
) -> PreprocessingResult:
    """Normalize, validate, filter, and audit one raw dataset split."""
    policy = policy or PreprocessingPolicy()
    accepted: list[PreparedSample] = []
    excluded: list[ExcludedRow] = []
    seen_ids: set[str] = set()
    content_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    raw_count = 0

    for source_index, row in enumerate(rows):
        raw_count += 1
        raw_id, raw_question, raw_options = _raw_preview(row)
        try:
            sample = normalizer(adapter, row, source_index)
        except Exception as exc:
            if policy.normalization_errors == "error":
                raise
            reason = "normalization_error"
            reason_counts[reason] += 1
            excluded.append(
                ExcludedRow(
                    source_index,
                    raw_id,
                    (reason,),
                    f"{type(exc).__name__}: {exc}",
                    raw_question,
                    raw_options,
                )
            )
            continue

        reasons: list[str] = []
        if sample.sample_id in seen_ids:
            if policy.duplicate_sample_ids == "error":
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            reasons.append("duplicate_sample_id")
        if policy.exclude_relative_position_references and any(
            has_relative_position_reference(option) for option in sample.options
        ):
            reasons.append("relative_position_reference")
        if policy.exclude_option_label_combinations and has_display_label_reference(
            sample.options
        ):
            reasons.append("option_label_combination")
        if policy.exclude_duplicate_option_texts and duplicate_option_texts(sample.options):
            reasons.append("duplicate_option_text")

        seen_ids.add(sample.sample_id)
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            excluded.append(
                ExcludedRow(
                    source_index,
                    sample.sample_id,
                    tuple(reasons),
                    None,
                    sample.question,
                    sample.options,
                )
            )
            continue

        content_hash = _content_hash(sample)
        content_counts[content_hash] += 1
        accepted.append(PreparedSample(source_index, sample, content_hash))

    duplicate_groups = sum(count > 1 for count in content_counts.values())
    duplicate_extra_rows = sum(count - 1 for count in content_counts.values() if count > 1)
    report = {
        "preprocessing_version": policy.version,
        "policy": policy.to_dict(),
        "policy_sha256": _sha256(_canonical_json(policy.to_dict())),
        "raw_rows": raw_count,
        "accepted_rows_before_selection": len(accepted),
        "excluded_rows": len(excluded),
        "excluded_by_reason": dict(sorted(reason_counts.items())),
        "exact_duplicate_content_groups_retained": duplicate_groups,
        "exact_duplicate_content_extra_rows_retained": duplicate_extra_rows,
        "accepted_sample_set_sha256": _sample_set_hash(accepted),
    }
    return PreprocessingResult(tuple(accepted), tuple(excluded), report)


def _samples_frame(prepared: Sequence[PreparedSample]) -> pd.DataFrame:
    rows = []
    for item in prepared:
        sample = item.sample
        rows.append(
            {
                "source_index": item.source_index,
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "subject": sample.subject,
                "question": sample.question,
                "option_A": sample.options[0],
                "option_B": sample.options[1],
                "option_C": sample.options[2],
                "option_D": sample.options[3],
                "gold_index": sample.gold_index,
                "gold_label": sample.gold_label,
                "content_sha256": item.content_hash,
            }
        )
    return pd.DataFrame.from_records(rows)


def _prompts_frame(
    prepared: Sequence[PreparedSample], prompt_builder: MCQPrompt
) -> pd.DataFrame:
    rows = []
    for item in prepared:
        for permutation in cyclic_permutations(item.sample):
            rows.append(
                {
                    "sample_id": item.sample.sample_id,
                    "permutation_id": permutation.permutation_id,
                    "display_to_content": list(permutation.display_to_content),
                    "content_to_display": list(permutation.content_to_display),
                    "displayed_option_A": permutation.displayed_options[0],
                    "displayed_option_B": permutation.displayed_options[1],
                    "displayed_option_C": permutation.displayed_options[2],
                    "displayed_option_D": permutation.displayed_options[3],
                    "gold_content_index": item.sample.gold_index,
                    "gold_displayed_label": permutation.gold_display_label,
                    "prompt_template_version": prompt_builder.config.version,
                    "user_prompt": prompt_builder.user_content(
                        item.sample.question, permutation.displayed_options
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_preprocessed_artifact(
    result: PreprocessingResult,
    output_dir: str | Path,
    *,
    prompt_builder: MCQPrompt,
    dataset_metadata: Mapping[str, Any],
) -> Path:
    """Write immutable, checksummed data and prompt artifacts for later runs."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite preprocessing artifact: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        samples_path = temporary / "samples.parquet"
        prompts_path = temporary / "cyclic_prompts.parquet"
        exclusions_path = temporary / "excluded_rows.parquet"
        _samples_frame(result.prepared).to_parquet(samples_path, index=False)
        _prompts_frame(result.prepared, prompt_builder).to_parquet(
            prompts_path, index=False
        )
        pd.DataFrame.from_records([row.to_dict() for row in result.excluded]).to_parquet(
            exclusions_path, index=False
        )

        manifest = {
            "schema_version": PREPROCESSED_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **dict(dataset_metadata),
            **result.report,
            "prompt_template_version": prompt_builder.config.version,
            "artifact_files": {
                path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
                for path in (samples_path, prompts_path, exclusions_path)
            },
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return root


def load_preprocessed_artifact(
    artifact_dir: str | Path,
    *,
    verify_hashes: bool = True,
) -> tuple[list[MCQSample], dict[str, Any]]:
    root = Path(artifact_dir)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != PREPROCESSED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported preprocessing schema: {manifest.get('schema_version')!r}"
        )
    if verify_hashes:
        for filename, metadata in manifest["artifact_files"].items():
            path = root / filename
            if not path.exists():
                raise FileNotFoundError(f"Preprocessing artifact file is missing: {path}")
            if _file_sha256(path) != metadata["sha256"]:
                raise ValueError(f"Checksum mismatch for preprocessing artifact: {path}")

    frame = pd.read_parquet(root / "samples.parquet")
    samples = [
        MCQSample(
            sample_id=str(row.sample_id),
            dataset=str(row.dataset),
            subject=None if pd.isna(row.subject) else str(row.subject),
            question=str(row.question),
            options=(
                str(row.option_A),
                str(row.option_B),
                str(row.option_C),
                str(row.option_D),
            ),
            gold_index=int(row.gold_index),
        )
        for row in frame.itertuples(index=False)
    ]
    return samples, manifest

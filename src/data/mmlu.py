from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import MCQSample, integer_label, text


def normalize_mmlu(row: Mapping[str, Any], row_index: int) -> MCQSample:
    raw_options = row.get("choices")
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        raise ValueError("MMLU choices must be a sequence")
    if len(raw_options) != 4:
        raise ValueError(f"MMLU row must contain four choices, got {len(raw_options)}")
    subject = text(row.get("subject", "unknown"), "subject")
    return MCQSample(
        sample_id=f"mmlu:{subject}:{row_index:06d}",
        dataset="mmlu",
        subject=subject,
        question=text(row.get("question"), "question"),
        options=tuple(text(value, f"choices[{i}]") for i, value in enumerate(raw_options)),  # type: ignore[arg-type]
        gold_index=integer_label(row.get("answer"), "answer"),
    )


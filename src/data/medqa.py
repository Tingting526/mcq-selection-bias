from __future__ import annotations

from typing import Any, Mapping

from .base import MCQSample, integer_label, text


def normalize_medqa(row: Mapping[str, Any], row_index: int) -> MCQSample:
    # sent2 is empty in the current HF conversion, but joining non-empty parts
    # preserves compatibility if a future pinned revision uses it.
    question_parts = [str(row.get(key, "")).strip() for key in ("sent1", "sent2")]
    question = "\n".join(part for part in question_parts if part)
    if not question:
        raise ValueError("MedQA row has neither sent1 nor sent2 content")
    options = tuple(text(row.get(f"ending{i}"), f"ending{i}") for i in range(4))
    return MCQSample(
        sample_id=text(row.get("id", f"medqa-{row_index:06d}"), "id"),
        dataset="medqa",
        question=question,
        options=options,  # type: ignore[arg-type]
        gold_index=integer_label(row.get("label"), "label"),
    )


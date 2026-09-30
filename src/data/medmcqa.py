from __future__ import annotations

from typing import Any, Mapping

from .base import MCQSample, integer_label, text


def normalize_medmcqa(row: Mapping[str, Any], row_index: int) -> MCQSample:
    options = tuple(text(row.get(key), key) for key in ("opa", "opb", "opc", "opd"))
    subject = row.get("subject_name")
    return MCQSample(
        sample_id=text(row.get("id", f"medmcqa-{row_index:06d}"), "id"),
        dataset="medmcqa",
        subject=str(subject).strip() if subject is not None and str(subject).strip() else None,
        question=text(row.get("question"), "question"),
        options=options,  # type: ignore[arg-type]
        gold_index=integer_label(row.get("cop"), "cop"),
    )


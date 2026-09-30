from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src import OPTION_LABELS


@dataclass(frozen=True, slots=True)
class MCQSample:
    sample_id: str
    dataset: str
    question: str
    options: tuple[str, str, str, str]
    gold_index: int
    subject: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.question.strip():
            raise ValueError(f"{self.sample_id}: question must not be empty")
        if len(self.options) != 4:
            raise ValueError(f"{self.sample_id}: exactly four options are required")
        if any(not isinstance(option, str) or not option.strip() for option in self.options):
            raise ValueError(f"{self.sample_id}: every option must be a non-empty string")
        if self.gold_index not in range(4):
            raise ValueError(f"{self.sample_id}: gold_index must be in 0..3")

    @property
    def gold_label(self) -> str:
        return OPTION_LABELS[self.gold_index]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["options"] = list(self.options)
        value["gold_label"] = self.gold_label
        return value


def text(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"Missing required field: {field}")
    result = str(value).strip()
    if not result:
        raise ValueError(f"Empty required field: {field}")
    return result


def integer_label(value: Any, field: str) -> int:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in OPTION_LABELS:
            return OPTION_LABELS.index(normalized)
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid label in {field}: {value!r}") from exc
    if label not in range(4):
        raise ValueError(f"Label in {field} must map to 0..3, got {label}")
    return label


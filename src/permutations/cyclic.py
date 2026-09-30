from __future__ import annotations

from dataclasses import dataclass

from src import OPTION_LABELS
from src.data.base import MCQSample


@dataclass(frozen=True, slots=True)
class CyclicPermutation:
    permutation_id: int
    display_to_content: tuple[int, int, int, int]
    content_to_display: tuple[int, int, int, int]
    displayed_options: tuple[str, str, str, str]
    gold_content_index: int
    gold_display_index: int

    @property
    def gold_display_label(self) -> str:
        return OPTION_LABELS[self.gold_display_index]

    def displayed_label_for_content(self, content_index: int) -> str:
        return OPTION_LABELS[self.content_to_display[content_index]]


def cyclic_permutations(sample: MCQSample) -> tuple[CyclicPermutation, ...]:
    result = []
    for shift in range(4):
        display_to_content = tuple((position + shift) % 4 for position in range(4))
        inverse = [0, 0, 0, 0]
        for display_index, content_index in enumerate(display_to_content):
            inverse[content_index] = display_index
        displayed_options = tuple(sample.options[index] for index in display_to_content)
        result.append(
            CyclicPermutation(
                permutation_id=shift,
                display_to_content=display_to_content,  # type: ignore[arg-type]
                content_to_display=tuple(inverse),  # type: ignore[arg-type]
                displayed_options=displayed_options,  # type: ignore[arg-type]
                gold_content_index=sample.gold_index,
                gold_display_index=inverse[sample.gold_index],
            )
        )
    return tuple(result)


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src import OPTION_LABELS


@dataclass(frozen=True, slots=True)
class PromptConfig:
    version: str
    instruction: str
    question_heading: str = "Question:"
    answer_heading: str = "Answer:"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PromptConfig":
        return cls(
            version=str(value["version"]),
            instruction=str(value["instruction"]),
            question_heading=str(value.get("question_heading", "Question:")),
            answer_heading=str(value.get("answer_heading", "Answer:")),
        )


class MCQPrompt:
    def __init__(self, config: PromptConfig):
        self.config = config

    def user_content(self, question: str, options: Sequence[str]) -> str:
        if len(options) != 4:
            raise ValueError("The canonical prompt requires exactly four options")
        option_lines = "\n".join(
            f"{label}. {option}" for label, option in zip(OPTION_LABELS, options, strict=True)
        )
        return (
            f"{self.config.instruction}\n\n"
            f"{self.config.question_heading} {question}\n\n"
            f"{option_lines}\n\n{self.config.answer_heading}"
        )

    def render(
        self,
        tokenizer: Any,
        question: str,
        options: Sequence[str],
        use_chat_template: bool,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        content = self.user_content(question, options)
        if not use_chat_template:
            return content
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "This model configuration requires a tokenizer chat template, but none is available. "
                "Set use_chat_template=false only after documenting the prompt change."
            )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            **dict(chat_template_kwargs or {}),
        )

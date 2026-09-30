from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from src import OPTION_LABELS
from src.data.base import MCQSample
from src.models.token_scoring import OptionScorer
from src.prompts.mcq import MCQPrompt


def _record(
    sample: MCQSample, model_name: str, score: Any, prompt: str, scorer_backend: str
) -> dict[str, Any]:
    predicted_label = score.predicted_label
    record: dict[str, Any] = {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "subject": sample.subject,
        "model": model_name,
        "scorer_backend": scorer_backend,
        "permutation_id": 0,
        "option_ordering": [0, 1, 2, 3],
        "options": list(sample.options),
        "correct_content_index": sample.gold_index,
        "correct_displayed_label": sample.gold_label,
        "predicted_label": predicted_label,
        "gold_label": sample.gold_label,
        "correct": predicted_label == sample.gold_label,
        "rendered_prompt": prompt,
    }
    for index, label in enumerate(OPTION_LABELS):
        record[f"logits_{label}"] = score.logits[index]
        record[f"prob_{label}"] = score.probabilities[index]
    return record


def iter_baseline_batches(
    samples: Sequence[MCQSample],
    *,
    scorer: OptionScorer,
    prompt_builder: MCQPrompt,
    tokenizer: Any,
    use_chat_template: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
    model_name: str,
    batch_size: int,
    completed_ids: set[str] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    completed = completed_ids or set()
    backend = getattr(scorer, "backend", "unknown")
    pending = [sample for sample in samples if sample.sample_id not in completed]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [
            prompt_builder.render(
                tokenizer,
                sample.question,
                sample.options,
                use_chat_template,
                chat_template_kwargs,
            )
            for sample in batch
        ]
        scores = scorer.score(prompts)
        yield [
            _record(sample, model_name, score, prompt, backend)
            for sample, score, prompt in zip(batch, scores, prompts, strict=True)
        ]

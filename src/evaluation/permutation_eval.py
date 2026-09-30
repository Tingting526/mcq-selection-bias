from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

from src import OPTION_LABELS
from src.data.base import MCQSample
from src.models.token_scoring import OptionScorer
from src.permutations.cyclic import cyclic_permutations
from src.pride.debias import cyclic_content_average
from src.prompts.mcq import MCQPrompt


def iter_permutation_batches(
    samples: Sequence[MCQSample],
    *,
    scorer: OptionScorer,
    prompt_builder: MCQPrompt,
    tokenizer: Any,
    use_chat_template: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
    model_name: str,
    batch_size: int,
    completed_keys: set[tuple[str, int]] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    work = []
    completed = completed_keys or set()
    backend = getattr(scorer, "backend", "unknown")
    for sample in samples:
        for permutation in cyclic_permutations(sample):
            if (sample.sample_id, permutation.permutation_id) not in completed:
                work.append((sample, permutation))
    for start in range(0, len(work), batch_size):
        batch = work[start : start + batch_size]
        prompts = [
            prompt_builder.render(
                tokenizer,
                sample.question,
                permutation.displayed_options,
                use_chat_template,
                chat_template_kwargs,
            )
            for sample, permutation in batch
        ]
        scores = scorer.score(prompts)
        records = []
        for (sample, permutation), score, prompt in zip(batch, scores, prompts, strict=True):
            record: dict[str, Any] = {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "subject": sample.subject,
                "model": model_name,
                "scorer_backend": backend,
                "permutation_id": permutation.permutation_id,
                "option_ordering": list(permutation.display_to_content),
                "options": list(permutation.displayed_options),
                "correct_content_index": sample.gold_index,
                "correct_displayed_label": permutation.gold_display_label,
                "predicted_label": score.predicted_label,
                "predicted_content_index": permutation.display_to_content[score.predicted_index],
                "gold_label": permutation.gold_display_label,
                "original_gold_label": sample.gold_label,
                "correct": score.predicted_label == permutation.gold_display_label,
                "rendered_prompt": prompt,
            }
            for index, label in enumerate(OPTION_LABELS):
                record[f"logits_{label}"] = score.logits[index]
                record[f"prob_{label}"] = score.probabilities[index]
            records.append(record)
        yield records


def cyclic_debiased_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    result = []
    for sample_id, sample_rows in grouped.items():
        ordered = sorted(sample_rows, key=lambda row: int(row["permutation_id"]))
        if [int(row["permutation_id"]) for row in ordered] != [0, 1, 2, 3]:
            raise ValueError(f"{sample_id}: all four cyclic permutations are required")
        probabilities = [[float(row[f"prob_{label}"]) for label in OPTION_LABELS] for row in ordered]
        mappings = [row["option_ordering"] for row in ordered]
        averaged = cyclic_content_average(probabilities, mappings)
        predicted_content = int(averaged.argmax())
        gold_content = int(ordered[0]["correct_content_index"])
        result.append(
            {
                "dataset": ordered[0]["dataset"],
                "sample_id": sample_id,
                "subject": ordered[0].get("subject"),
                "model": ordered[0]["model"],
                "predicted_content_index": predicted_content,
                "gold_content_index": gold_content,
                "predicted_label": OPTION_LABELS[predicted_content],
                "gold_label": OPTION_LABELS[gold_content],
                "correct": predicted_content == gold_content,
                **{f"cyclic_prob_content_{label}": float(averaged[i]) for i, label in enumerate(OPTION_LABELS)},
            }
        )
    return result

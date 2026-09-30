from __future__ import annotations

import argparse
import math
import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT
from src.data.base import MCQSample
from src.models.loader import load_model
from src.models.token_scoring import (
    FirstTokenOptionScorer,
    MockOptionScorer,
    OptionScorer,
    contextual_option_tokens,
    contextual_token_ids,
    diagnostic_to_dict,
)
from src.prompts.mcq import MCQPrompt, PromptConfig
from src.utils.config import load_yaml, named_config
from src.utils.io import RunStore, unique_path, write_json  # noqa: F401
from src.utils.logging import configure_logging
from src.utils.seeds import set_deterministic_seed


@dataclass(slots=True)
class Runtime:
    scorer: OptionScorer
    scorer_backend: str
    prompt_builder: MCQPrompt
    experiment: dict[str, Any]
    tokenizer: Any
    use_chat_template: bool
    chat_template_kwargs: dict[str, Any]
    model_name: str
    rendered_diagnostic_prompt: str
    resolved_model_revision: str | None = None
    resolved_tokenizer_revision: str | None = None
    resolved_device: Any = None
    model_class_name: str | None = None
    tokenizer_class_name: str | None = None
    option_token_ids: list[int] | None = None
    token_diagnostic: dict[str, Any] | None = None


def add_common_arguments(parser: argparse.ArgumentParser, *, include_dataset: bool = True) -> None:
    parser.add_argument("--model", required=True, help="Alias from configs/models")
    if include_dataset:
        parser.add_argument("--dataset", required=True, help="Alias from configs/datasets")
        parser.add_argument(
            "--preprocessed",
            type=Path,
            help="Audited artifact directory produced by scripts/preprocess_data.py",
        )
    parser.add_argument(
        "--split",
        help="Dataset split. If omitted, use default_split from the dataset configuration.",
    )
    parser.add_argument("--experiment", default="configs/experiments/thesis.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--scorer",
        choices=("real", "mock"),
        default="real",
        help="real = load the model and use its logits (default, thesis path); "
        "mock = deterministic model-free scoring for CPU pipeline testing only.",
    )
    parser.add_argument("--verbose", action="store_true")


def resolved_device(model: Any) -> Any:
    device_map = getattr(model, "hf_device_map", None)
    return device_map if device_map is not None else str(getattr(model, "device", "unknown"))


def _save_token_diagnostic(model_alias: str, rendered_prompt: str, diagnostic: dict[str, Any]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = PROJECT_ROOT / "outputs" / "logs" / "token_diagnostics" / f"{timestamp}_{model_alias}.json"
    payload = {
        "model_alias": model_alias,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rendered_prompt_ending": rendered_prompt[-800:],
        **diagnostic,
    }
    write_json(path, payload)
    return path


def load_runtime(
    model_alias: str,
    experiment_path: str,
    sample: MCQSample,
    seed_override: int | None,
    *,
    scorer_backend: str = "real",
) -> Runtime:
    experiment = load_yaml(experiment_path)
    seed = int(seed_override if seed_override is not None else experiment["seed"])
    set_deterministic_seed(seed)
    model_config = named_config("models", model_alias)
    prompt_builder = MCQPrompt(PromptConfig.from_mapping(experiment["prompt"]))
    model_name = str(model_config["model_repo"])

    if scorer_backend == "mock":
        print("Scorer backend: MOCK (no model loaded). For CPU testing only - not thesis results.")
        rendered = prompt_builder.user_content(sample.question, sample.options)
        return Runtime(
            scorer=MockOptionScorer(seed=seed),
            scorer_backend="mock",
            prompt_builder=prompt_builder,
            experiment=experiment,
            tokenizer=None,
            use_chat_template=False,
            chat_template_kwargs={},
            model_name=model_name,
            rendered_diagnostic_prompt=rendered,
            model_class_name="mock",
            tokenizer_class_name="mock",
        )

    # --- real backend. load_model runs the hardware capacity gate BEFORE download. ---
    # Any failure here propagates: we NEVER silently fall back to the mock scorer.
    loaded = load_model(model_config)
    use_chat_template = bool(model_config.get("use_chat_template", True))
    chat_template_kwargs = dict(model_config.get("chat_template_kwargs") or {})
    rendered = prompt_builder.render(
        loaded.tokenizer,
        sample.question,
        sample.options,
        use_chat_template,
        chat_template_kwargs,
    )

    # Contextual A/B/C/D diagnostic against the EXACT rendered prompt (fails loudly).
    add_special_tokens = not use_chat_template
    rows = contextual_option_tokens(
        loaded.tokenizer,
        rendered,
        add_special_tokens=add_special_tokens,
    )
    token_ids = contextual_token_ids(rows)
    diagnostic = diagnostic_to_dict(rows)
    diag_path = _save_token_diagnostic(model_alias, rendered, diagnostic)

    print("Contextual option-token diagnostic passed:")
    print("Rendered prompt ending:")
    print(rendered[-200:])
    for row in rows:
        print(
            f"  candidate {row.label}: continuation={row.continuation_repr!r} "
            f"token_id={row.token_id} decoded={row.decoded_token!r}"
        )
    print(f"Saved token diagnostic: {diag_path}")

    scorer = FirstTokenOptionScorer(
        loaded.model,
        loaded.tokenizer,
        token_ids,
        add_special_tokens=add_special_tokens,
    )
    return Runtime(
        scorer=scorer,
        scorer_backend=scorer.backend,
        prompt_builder=prompt_builder,
        experiment=experiment,
        tokenizer=loaded.tokenizer,
        use_chat_template=use_chat_template,
        chat_template_kwargs=chat_template_kwargs,
        model_name=model_name,
        rendered_diagnostic_prompt=rendered,
        resolved_model_revision=loaded.resolved_model_revision,
        resolved_tokenizer_revision=loaded.resolved_tokenizer_revision,
        resolved_device=resolved_device(loaded.model),
        model_class_name=loaded.model_class_name,
        tokenizer_class_name=loaded.tokenizer_class_name,
        option_token_ids=token_ids,
        token_diagnostic=diagnostic,
    )


def effective_batch_size(args: argparse.Namespace, experiment: dict[str, Any]) -> int:
    value = int(args.batch_size if args.batch_size is not None else experiment["batch_size"])
    if value <= 0:
        raise ValueError("batch_size must be positive")
    return value


def effective_seed(args: argparse.Namespace, experiment: dict[str, Any]) -> int:
    return int(args.seed if args.seed is not None else experiment["seed"])


def effective_split(requested: str | None, dataset_config: dict[str, Any]) -> str:
    split = requested or dataset_config.get("default_split")
    if not split:
        raise ValueError("A split must be provided either by --split or dataset default_split")
    return str(split)


def collect_batches(
    batches: Iterator[list[dict[str, Any]]],
    *,
    store: RunStore,
    total_items: int,
    description: str,
    flush_every_batches: int,
) -> None:
    pending: list[dict[str, Any]] = []
    debug_prompts: list[dict[str, Any]] = []
    progress = tqdm(total=total_items, desc=description, unit="prompt")
    try:
        for batch_index, batch in enumerate(batches, start=1):
            for row in batch:
                rendered = row.pop("rendered_prompt", None)
                if rendered is not None and len(debug_prompts) < 3:
                    debug_prompts.append(
                        {
                            "sample_id": row["sample_id"],
                            "permutation_id": row.get("permutation_id"),
                            "rendered_prompt": rendered,
                        }
                    )
            pending.extend(batch)
            progress.update(len(batch))
            if batch_index % flush_every_batches == 0:
                store.write_records(pending)
                pending = []
    finally:
        if pending:
            store.write_records(pending)
        progress.close()
        if debug_prompts:
            store.save_debug_prompts(debug_prompts)


def print_forward_pass_estimate(prompt_count: int, batch_size: int) -> None:
    print(f"Planned prompts: {prompt_count:,}")
    print(f"Estimated batched forward passes: {math.ceil(prompt_count / batch_size):,}")


def save_summary(store: RunStore, metrics: dict[str, Any], stem: str) -> tuple[Path, Path]:
    metadata = json.loads((store.run_dir / "metadata.json").read_text())
    metrics = {
        **metrics,
        "pipeline_id": os.environ.get("MCQ_PIPELINE_ID", "unmanaged"),
        "run_dir": str(store.run_dir.resolve()),
        "kind": store.run_dir.parent.name,
        "model_repo": metadata["model_repo"],
        "dataset_repo": metadata["dataset_repo"],
        "scorer_backend": metadata["scorer_backend"],
        "seed": metadata["seed"],
        "selection_limit": metadata["selection_limit"],
        "sample_set_sha256": metadata["preprocessing"].get("analysis_sample_set_sha256"),
    }
    local_path = unique_path(store.run_dir, "summary", ".json")
    write_json(local_path, metrics)
    global_path = unique_path(PROJECT_ROOT / "outputs" / "summaries", stem, ".csv")
    pd.DataFrame([metrics]).to_csv(global_path, index=False)
    return local_path, global_path


def configure_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.verbose)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")

from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import PROJECT_ROOT
from _common import (
    add_common_arguments,
    collect_batches,
    configure_from_args,
    effective_batch_size,
    effective_seed,
    effective_split,
    load_runtime,
    print_forward_pass_estimate,
    save_summary,
)
from src.data.registry import load_samples_with_metadata
from src.evaluation.permutation_eval import cyclic_debiased_records, iter_permutation_batches
from src.metrics.selection_bias import permutation_accuracy_metrics, selection_bias_metrics
from src.utils.config import named_config
from src.utils.io import RunStore, unique_path
from src.utils.metadata import run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all four cyclic option permutations")
    add_common_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    dataset_config = named_config("datasets", args.dataset)
    model_config = named_config("models", args.model)
    split = effective_split(args.split, dataset_config)
    samples, fingerprint, preprocessing = load_samples_with_metadata(
        dataset_config,
        split,
        limit=args.limit,
        start_index=args.start_index,
        preprocessed=args.preprocessed,
    )
    if not samples:
        raise ValueError("No samples selected")
    runtime = load_runtime(args.model, args.experiment, samples[0], args.seed, scorer_backend=args.scorer)
    batch_size = effective_batch_size(args, runtime.experiment)
    seed = effective_seed(args, runtime.experiment)
    store = RunStore.create("permutations", args.model, args.dataset, resume=args.resume)
    completed = store.completed_permutations()
    remaining = sum(
        (sample.sample_id, permutation_id) not in completed
        for sample in samples
        for permutation_id in range(4)
    )
    print_forward_pass_estimate(remaining, batch_size)
    store.save_metadata(
        run_metadata(
            model_config=model_config,
            dataset_config=dataset_config,
            split=split,
            seed=seed,
            batch_size=batch_size,
            prompt_template_version=runtime.experiment["prompt"]["version"],
            dataset_fingerprint=fingerprint,
            resolved_model_revision=runtime.resolved_model_revision,
            resolved_tokenizer_revision=runtime.resolved_tokenizer_revision,
            resolved_device=runtime.resolved_device,
            start_index=args.start_index,
            limit=args.limit,
            scorer_backend=runtime.scorer_backend,
            model_class=runtime.model_class_name,
            tokenizer_class=runtime.tokenizer_class_name,
            option_token_ids=runtime.option_token_ids,
            preprocessing_metadata=preprocessing,
        )
    )
    batches = iter_permutation_batches(
        samples,
        scorer=runtime.scorer,
        prompt_builder=runtime.prompt_builder,
        tokenizer=runtime.tokenizer,
        use_chat_template=runtime.use_chat_template,
        chat_template_kwargs=runtime.chat_template_kwargs,
        model_name=model_config["model_repo"],
        batch_size=batch_size,
        completed_keys=completed,
    )
    collect_batches(
        batches,
        store=store,
        total_items=remaining,
        description="Cyclic permutations",
        flush_every_batches=int(runtime.experiment["flush_every_batches"]),
    )
    rows = store.read_records()
    cyclic_rows = cyclic_debiased_records(rows)
    cyclic_path = unique_path(store.run_dir, "cyclic_debiased", ".parquet")
    pd.DataFrame(cyclic_rows).to_parquet(cyclic_path, index=False)
    metrics = selection_bias_metrics(rows)
    metrics.update(permutation_accuracy_metrics(rows))
    cyclic_metrics = selection_bias_metrics(cyclic_rows)
    metrics.update({f"cyclic_debiased_{key}": value for key, value in cyclic_metrics.items()})
    _, summary_path = save_summary(store, metrics, f"permutations_{args.model}_{args.dataset}")
    print(f"Raw results: {store.run_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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
from src.evaluation.baseline import iter_baseline_batches
from src.evaluation.permutation_eval import cyclic_debiased_records, iter_permutation_batches
from src.evaluation.pride_eval import (
    apply_prior_to_records,
    estimate_prior_from_permutation_records,
    split_estimation_and_remaining,
)
from src.metrics.selection_bias import selection_bias_metrics
from src.pride.prior_estimation import select_estimation_ids
from src.pride.transfer import transfer_summary
from src.utils.config import load_yaml, named_config
from src.utils.io import RunStore, unique_path, write_json
from src.utils.metadata import run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate and transfer a PriDe prior between datasets")
    add_common_arguments(parser, include_dataset=False)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--source-limit", type=int)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--source-preprocessed", type=Path)
    parser.add_argument("--target-preprocessed", type=Path)
    args = parser.parse_args()
    configure_from_args(args)

    source_config = named_config("datasets", args.source)
    target_config = named_config("datasets", args.target)
    model_config = named_config("models", args.model)
    source_split = effective_split(args.split, source_config)
    target_split = effective_split(args.split, target_config)
    source_limit = args.source_limit if args.source_limit is not None else args.limit
    target_limit = args.target_limit if args.target_limit is not None else args.limit
    source_samples, source_fingerprint, source_preprocessing = load_samples_with_metadata(
        source_config,
        source_split,
        limit=source_limit,
        start_index=args.start_index,
        preprocessed=args.source_preprocessed,
    )
    target_samples, target_fingerprint, target_preprocessing = load_samples_with_metadata(
        target_config,
        target_split,
        limit=target_limit,
        start_index=args.start_index,
        preprocessed=args.target_preprocessed,
    )
    if not source_samples or not target_samples:
        raise ValueError("Both source and target must contain samples")
    experiment = load_yaml(args.experiment)
    pride_config = experiment["pride"]
    alpha = float(args.alpha if args.alpha is not None else pride_config["alpha"])
    seed = int(args.seed if args.seed is not None else experiment["seed"])
    estimation_ids = select_estimation_ids(
        (sample.sample_id for sample in source_samples),
        alpha,
        seed,
        rounding=str(pride_config["alpha_rounding"]),
        minimum=int(pride_config["minimum_estimation_samples"]),
        sort_before_sampling=str(pride_config["sampling_order"]) == "sorted_sample_id",
    )
    estimation_set = set(estimation_ids)
    estimation_samples = [sample for sample in source_samples if sample.sample_id in estimation_set]
    runtime = load_runtime(
        args.model, args.experiment, estimation_samples[0], args.seed, scorer_backend=args.scorer
    )
    batch_size = effective_batch_size(args, runtime.experiment)

    parent = RunStore.create(
        "transfer", args.model, f"{args.source}_to_{args.target}", resume=args.resume
    )
    metadata = run_metadata(
        model_config=model_config,
        dataset_config=source_config,
        split=source_split,
        seed=effective_seed(args, runtime.experiment),
        batch_size=batch_size,
        prompt_template_version=runtime.experiment["prompt"]["version"],
        dataset_fingerprint=source_fingerprint,
        resolved_model_revision=runtime.resolved_model_revision,
        resolved_tokenizer_revision=runtime.resolved_tokenizer_revision,
        resolved_device=runtime.resolved_device,
        start_index=args.start_index,
        limit=source_limit,
        alpha=alpha,
        estimation_ids=estimation_ids,
        scorer_backend=runtime.scorer_backend,
        model_class=runtime.model_class_name,
        tokenizer_class=runtime.tokenizer_class_name,
        option_token_ids=runtime.option_token_ids,
        preprocessing_metadata=source_preprocessing,
    )
    metadata.update(
        target_dataset_repo=target_config["repo"],
        target_dataset_revision=target_config.get("revision"),
        target_dataset_fingerprint=target_fingerprint,
        target_dataset_split=target_split,
        target_selection_limit=target_limit,
        target_preprocessing=target_preprocessing,
    )
    parent.save_metadata(metadata)

    source_store = RunStore(parent.run_dir / "source_permutations")
    source_completed = source_store.completed_permutations()
    source_remaining = sum(
        (sample.sample_id, permutation_id) not in source_completed
        for sample in estimation_samples
        for permutation_id in range(4)
    )
    print("Source prior estimation")
    print_forward_pass_estimate(source_remaining, batch_size)
    source_batches = iter_permutation_batches(
        estimation_samples,
        scorer=runtime.scorer,
        prompt_builder=runtime.prompt_builder,
        tokenizer=runtime.tokenizer,
        use_chat_template=runtime.use_chat_template,
        chat_template_kwargs=runtime.chat_template_kwargs,
        model_name=model_config["model_repo"],
        batch_size=batch_size,
        completed_keys=source_completed,
    )
    collect_batches(
        source_batches,
        store=source_store,
        total_items=source_remaining,
        description="Transfer source",
        flush_every_batches=int(runtime.experiment["flush_every_batches"]),
    )
    source_rows = source_store.read_records()
    _, global_prior = estimate_prior_from_permutation_records(
        source_rows, epsilon=float(pride_config["epsilon"])
    )
    cyclic_source_rows = cyclic_debiased_records(source_rows)
    prior_payload = {
        "method": "PriDe (Zheng et al., 2024)",
        "source_dataset": args.source,
        "target_dataset": args.target,
        "model_repo": model_config["model_repo"],
        "alpha": alpha,
        "seed": seed,
        "epsilon": float(pride_config["epsilon"]),
        "global_prior": global_prior,
        "estimation_sample_ids": estimation_ids,
        "estimation_sample_policy": pride_config["estimation_sample_policy"],
        "prompt_template_version": runtime.experiment["prompt"]["version"],
        "chat_template_kwargs": runtime.chat_template_kwargs,
        "source_preprocessing": source_preprocessing,
        "source_analysis_sample_set_sha256": source_preprocessing[
            "analysis_sample_set_sha256"
        ],
    }
    prior_path = unique_path(parent.run_dir, "prior", ".json")
    write_json(prior_path, prior_payload)

    target_store = RunStore(parent.run_dir / "target_baseline")
    if args.source == args.target and pride_config["estimation_sample_policy"] == "cyclic":
        target_ids = {sample.sample_id for sample in target_samples}
        already_completed = target_store.completed_ids()
        reusable_defaults = [
            row
            for row in source_rows
            if int(row["permutation_id"]) == 0
            and str(row["sample_id"]) in target_ids
            and str(row["sample_id"]) not in already_completed
        ]
        target_store.write_records(reusable_defaults)
    target_completed = target_store.completed_ids()
    target_remaining = sum(sample.sample_id not in target_completed for sample in target_samples)
    print("Target baseline and PriDe correction")
    print_forward_pass_estimate(target_remaining, batch_size)
    target_batches = iter_baseline_batches(
        target_samples,
        scorer=runtime.scorer,
        prompt_builder=runtime.prompt_builder,
        tokenizer=runtime.tokenizer,
        use_chat_template=runtime.use_chat_template,
        chat_template_kwargs=runtime.chat_template_kwargs,
        model_name=model_config["model_repo"],
        batch_size=batch_size,
        completed_ids=target_completed,
    )
    collect_batches(
        target_batches,
        store=target_store,
        total_items=target_remaining,
        description="Transfer target",
        flush_every_batches=int(runtime.experiment["flush_every_batches"]),
    )
    baseline_rows = target_store.read_records()
    pride_rows = apply_prior_to_records(
        baseline_rows, global_prior, epsilon=float(pride_config["epsilon"])
    )
    if args.source == args.target and pride_config["estimation_sample_policy"] == "cyclic":
        pride_rows = split_estimation_and_remaining(
            pride_rows, cyclic_source_rows, estimation_ids, transferred=False
        )
    else:
        pride_rows = split_estimation_and_remaining(
            pride_rows, [], estimation_ids, transferred=True
        )
    pride_path = unique_path(parent.run_dir, "target_pride_results", ".parquet")
    pd.DataFrame(pride_rows).to_parquet(pride_path, index=False)

    baseline_metrics = selection_bias_metrics(baseline_rows)
    pride_metrics = selection_bias_metrics(pride_rows)
    summary = transfer_summary(
        source_dataset=args.source,
        target_dataset=args.target,
        model=args.model,
        alpha=alpha,
        prior=global_prior,
        baseline=baseline_metrics,
        pride=pride_metrics,
    )
    _, summary_path = save_summary(
        parent, summary, f"transfer_{args.model}_{args.source}_to_{args.target}"
    )
    print(f"Prior: {prior_path}")
    print(f"PriDe target results: {pride_path}")
    print(f"Transfer summary: {summary_path}")


if __name__ == "__main__":
    main()

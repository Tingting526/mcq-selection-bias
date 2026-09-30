from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

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
)
from src.data.registry import load_samples_with_metadata
from src.evaluation.permutation_eval import cyclic_debiased_records, iter_permutation_batches
from src.evaluation.pride_eval import estimate_prior_from_permutation_records
from src.pride.prior_estimation import select_estimation_ids
from src.utils.config import named_config
from src.utils.io import RunStore, unique_path, write_json
from src.utils.metadata import run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a label-free PriDe global prior")
    add_common_arguments(parser)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--output", type=Path, help="Optional exact JSON output path")
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
    experiment_path = args.experiment
    from src.utils.config import load_yaml

    experiment = load_yaml(experiment_path)
    pride_config = experiment["pride"]
    alpha = float(args.alpha if args.alpha is not None else pride_config["alpha"])
    seed = int(args.seed if args.seed is not None else experiment["seed"])
    estimation_ids = select_estimation_ids(
        (sample.sample_id for sample in samples),
        alpha,
        seed,
        rounding=str(pride_config["alpha_rounding"]),
        minimum=int(pride_config["minimum_estimation_samples"]),
        sort_before_sampling=str(pride_config["sampling_order"]) == "sorted_sample_id",
    )
    selected_ids = set(estimation_ids)
    selected_samples = [sample for sample in samples if sample.sample_id in selected_ids]
    runtime = load_runtime(
        args.model, args.experiment, selected_samples[0], args.seed, scorer_backend=args.scorer
    )
    batch_size = effective_batch_size(args, runtime.experiment)
    store = RunStore.create("prior-estimation", args.model, args.dataset, resume=args.resume)
    completed = store.completed_permutations()
    remaining = sum(
        (sample.sample_id, permutation_id) not in completed
        for sample in selected_samples
        for permutation_id in range(4)
    )
    print(f"Estimation set: {len(selected_samples):,} of {len(samples):,} samples (alpha={alpha:g})")
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
            alpha=alpha,
            estimation_ids=estimation_ids,
            scorer_backend=runtime.scorer_backend,
            model_class=runtime.model_class_name,
            tokenizer_class=runtime.tokenizer_class_name,
            option_token_ids=runtime.option_token_ids,
            preprocessing_metadata=preprocessing,
        )
    )
    batches = iter_permutation_batches(
        selected_samples,
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
        description="PriDe prior estimation",
        flush_every_batches=int(runtime.experiment["flush_every_batches"]),
    )
    raw_rows = store.read_records()
    epsilon = float(pride_config["epsilon"])
    sample_priors, global_prior = estimate_prior_from_permutation_records(
        raw_rows, epsilon=epsilon
    )
    sample_prior_path = unique_path(store.run_dir, "sample_priors", ".parquet")
    pd.DataFrame(sample_priors).to_parquet(sample_prior_path, index=False)
    cyclic_path = unique_path(store.run_dir, "estimation_cyclic_predictions", ".parquet")
    pd.DataFrame(cyclic_debiased_records(raw_rows)).to_parquet(cyclic_path, index=False)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output is not None:
        prior_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    else:
        prior_path = unique_path(
            PROJECT_ROOT / "outputs" / "priors",
            f"{timestamp}_{args.model}_{args.dataset}_alpha-{alpha:g}",
            ".json",
        )
    payload = {
        "method": "PriDe (Zheng et al., 2024)",
        "scorer_backend": runtime.scorer_backend,
        "source_dataset": args.dataset,
        "dataset_repo": dataset_config["repo"],
        "dataset_split": split,
        "model_alias": args.model,
        "model_repo": model_config["model_repo"],
        "alpha": alpha,
        "seed": seed,
        "epsilon": epsilon,
        "global_prior": global_prior,
        "estimation_sample_ids": estimation_ids,
        "alpha_rounding": pride_config["alpha_rounding"],
        "sampling_order": pride_config["sampling_order"],
        "estimation_sample_policy": pride_config["estimation_sample_policy"],
        "prompt_template_version": runtime.experiment["prompt"]["version"],
        "chat_template_kwargs": runtime.chat_template_kwargs,
        "raw_run_directory": str(store.run_dir.resolve()),
        "sample_priors_path": str(sample_prior_path.resolve()),
        "estimation_cyclic_predictions_path": str(cyclic_path.resolve()),
        "preprocessing": preprocessing,
        "analysis_sample_set_sha256": preprocessing["analysis_sample_set_sha256"],
    }
    write_json(prior_path, payload)
    print(f"Global prior A/B/C/D: {[round(value, 8) for value in global_prior]}")
    print(f"Prior: {prior_path}")
    print(f"Raw estimation results: {store.run_dir}")


if __name__ == "__main__":
    main()

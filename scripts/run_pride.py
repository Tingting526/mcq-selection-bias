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
from src import OPTION_LABELS
from src.data.registry import load_samples_with_metadata
from src.evaluation.baseline import iter_baseline_batches
from src.evaluation.pride_eval import apply_prior_to_records, split_estimation_and_remaining
from src.metrics.selection_bias import selection_bias_metrics
from src.utils.config import named_config, load_yaml
from src.evaluation.saved_baseline import reuse_saved_baseline
from src.utils.io import RunStore, read_json, unique_path
from src.utils.metadata import run_metadata


def _load_cyclic_rows(prior_payload: dict) -> list[dict]:
    """Cyclic (D_e) predictions saved during prior estimation, if the policy uses them."""
    if prior_payload.get("estimation_sample_policy") != "cyclic":
        return []
    path = Path(prior_payload["estimation_cyclic_predictions_path"])
    if not path.exists():
        raise FileNotFoundError(f"Cyclic estimation predictions referenced by the prior are missing: {path}")
    return pd.read_parquet(path).to_dict(orient="records")


def _reuse_estimation_defaults(
    store: RunStore,
    prior_payload: dict,
    target_ids: set[str],
) -> None:
    if prior_payload.get("estimation_sample_policy") != "cyclic":
        return
    source_path = Path(prior_payload["raw_run_directory"])
    if not source_path.exists():
        raise FileNotFoundError(f"Raw prior-estimation run is missing: {source_path}")
    source_store = RunStore(source_path)
    already_completed = store.completed_ids()
    reusable = [
        row
        for row in source_store.read_records()
        if int(row["permutation_id"]) == 0
        and str(row["sample_id"]) in target_ids
        and str(row["sample_id"]) not in already_completed
    ]
    store.write_records(reusable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a saved PriDe prior to default-order questions")
    add_common_arguments(parser)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, help="Reuse original predictions for a paired comparison")
    args = parser.parse_args()
    configure_from_args(args)

    prior_payload = read_json(args.prior)
    expected_backend = "mock" if args.scorer == "mock" else "real_transformers"
    if prior_payload.get("scorer_backend") != expected_backend:
        raise ValueError("Prior scorer backend does not match requested backend")
    dataset_config = named_config("datasets", args.dataset)
    model_config = named_config("models", args.model)
    split = effective_split(args.split, dataset_config)
    if prior_payload["model_repo"] != model_config["model_repo"]:
        raise ValueError("The saved prior was estimated with a different model repository")
    samples, fingerprint, preprocessing = load_samples_with_metadata(
        dataset_config,
        split,
        limit=args.limit,
        start_index=args.start_index,
        preprocessed=args.preprocessed,
    )
    if not samples:
        raise ValueError("No samples selected")
    if (
        args.dataset == prior_payload["source_dataset"]
        and prior_payload.get("analysis_sample_set_sha256")
        != preprocessing["analysis_sample_set_sha256"]
    ):
        raise ValueError(
            "The saved prior was estimated from a different preprocessed sample population"
        )
    if args.baseline_run:
        experiment = load_yaml(args.experiment)
        store = RunStore.create("pride", args.model, args.dataset, resume=args.resume)
        reuse_saved_baseline(
            args.baseline_run, store, prior_payload, samples,
            expected={
                "scorer_backend": expected_backend,
                "model_repo": model_config["model_repo"],
                "dataset_repo": dataset_config["repo"],
                "dataset_split": split,
                "selection_limit": args.limit,
                "selection_start_index": args.start_index,
                "seed": effective_seed(args, experiment),
                "prompt_template_version": experiment["prompt"]["version"],
                "preprocessing": preprocessing,
            },
        )
    else:
        runtime = load_runtime(args.model, args.experiment, samples[0], args.seed, scorer_backend=args.scorer)
        if prior_payload["prompt_template_version"] != runtime.experiment["prompt"]["version"]:
            raise ValueError("The saved prior used a different prompt template version")
        if dict(prior_payload.get("chat_template_kwargs") or {}) != runtime.chat_template_kwargs:
            raise ValueError("The saved prior used different model chat-template settings")
        batch_size = effective_batch_size(args, runtime.experiment)
        seed = effective_seed(args, runtime.experiment)
        store = RunStore.create("pride", args.model, args.dataset, resume=args.resume)
        metadata = run_metadata(
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
            alpha=float(prior_payload["alpha"]),
            estimation_ids=list(prior_payload["estimation_sample_ids"]),
            scorer_backend=runtime.scorer_backend,
            model_class=runtime.model_class_name,
            tokenizer_class=runtime.tokenizer_class_name,
            option_token_ids=runtime.option_token_ids,
            preprocessing_metadata=preprocessing,
        )
        metadata["prior_source_dataset"] = prior_payload["source_dataset"]
        metadata["prior_global"] = prior_payload["global_prior"]
        store.save_metadata(metadata)
        if args.dataset == prior_payload["source_dataset"]:
            _reuse_estimation_defaults(
                store, prior_payload, {sample.sample_id for sample in samples}
            )
        completed = store.completed_ids()
        remaining = sum(sample.sample_id not in completed for sample in samples)
        print_forward_pass_estimate(remaining, batch_size)
        batches = iter_baseline_batches(
            samples,
            scorer=runtime.scorer,
            prompt_builder=runtime.prompt_builder,
            tokenizer=runtime.tokenizer,
            use_chat_template=runtime.use_chat_template,
            chat_template_kwargs=runtime.chat_template_kwargs,
            model_name=model_config["model_repo"],
            batch_size=batch_size,
            completed_ids=completed,
        )
        collect_batches(
            batches,
            store=store,
            total_items=remaining,
            description="PriDe target baseline",
            flush_every_batches=int(runtime.experiment["flush_every_batches"]),
        )
    baseline_rows = store.read_records()
    pride_rows = apply_prior_to_records(
        baseline_rows,
        prior_payload["global_prior"],
        epsilon=float(prior_payload["epsilon"]),
    )
    estimation_ids = list(prior_payload["estimation_sample_ids"])
    if args.dataset == prior_payload["source_dataset"]:
        pride_rows = split_estimation_and_remaining(
            pride_rows, _load_cyclic_rows(prior_payload), estimation_ids, transferred=False
        )
    else:
        pride_rows = split_estimation_and_remaining(
            pride_rows, [], estimation_ids, transferred=True
        )
    output_path = unique_path(store.run_dir, "pride_results", ".parquet")
    pd.DataFrame(pride_rows).to_parquet(output_path, index=False)
    baseline_metrics = selection_bias_metrics(baseline_rows)
    pride_metrics = selection_bias_metrics(pride_rows)
    metrics = {
        "source_dataset": prior_payload["source_dataset"],
        "target_dataset": args.dataset,
        "alpha": prior_payload["alpha"],
        **{f"prior_{label}": prior_payload["global_prior"][i] for i, label in enumerate(OPTION_LABELS)},
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        **{f"pride_{key}": value for key, value in pride_metrics.items()},
        "delta_accuracy": pride_metrics["accuracy"] - baseline_metrics["accuracy"],
        "delta_rstd": pride_metrics["rstd"] - baseline_metrics["rstd"],
    }
    _, summary_path = save_summary(store, metrics, f"pride_{args.model}_{args.dataset}")
    print(f"PriDe results: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

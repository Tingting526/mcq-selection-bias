import pytest

from src.utils.io import RunStore, read_json


def test_run_store_writes_parquet_and_recovers_completed_keys(tmp_path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_records(
        [
            {"sample_id": "q1", "permutation_id": 0, "prob_A": 0.4},
            {"sample_id": "q1", "permutation_id": 1, "prob_A": 0.3},
        ]
    )
    assert store.completed_ids() == {"q1"}
    assert store.completed_permutations() == {("q1", 0), ("q1", 1)}
    assert len(store.read_records()) == 2


def test_resume_metadata_rejects_methodological_mismatch(tmp_path) -> None:
    store = RunStore(tmp_path / "run")
    metadata = {
        "model_repo": "model",
        "model_revision_requested": "abc",
        "tokenizer_repo": "model",
        "dataset_repo": "dataset",
        "dataset_revision": "def",
        "dataset_split": "test",
        "seed": 42,
        "prompt_template_version": "v1",
        "estimation_fraction_alpha": 0.05,
        "prior_estimation_sample_ids": ["q1"],
    }
    path = store.save_metadata(metadata)
    assert read_json(path)["seed"] == 42
    changed = dict(metadata, seed=7)
    with pytest.raises(ValueError, match="seed"):
        store.save_metadata(changed)


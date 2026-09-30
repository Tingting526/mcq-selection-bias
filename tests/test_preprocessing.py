import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.base import MCQSample
from src.data.preprocessing import (
    PREPROCESSED_SCHEMA_VERSION,
    PreprocessingPolicy,
    has_display_label_reference,
    has_option_label_combination,
    has_relative_position_reference,
    load_preprocessed_artifact,
    preprocess_rows,
    write_preprocessed_artifact,
)
from src.data.registry import normalize_row
from src.data.registry import load_samples_with_metadata
from src.prompts.mcq import MCQPrompt, PromptConfig


def _mmlu(question, choices, answer=0, subject="test"):
    return {"question": question, "choices": choices, "answer": answer, "subject": subject}


def test_permutation_unsafe_option_detectors_avoid_obvious_false_positives():
    assert has_relative_position_reference("None of the above.")
    assert has_relative_position_reference("both of above")
    assert has_option_label_combination("Both A and C.")
    assert has_option_label_combination("A B and C")
    assert not has_option_label_combination("Vitamin A and D")
    assert not has_option_label_combination("Countries A and B have equal income")
    assert has_display_label_reference(["one", "two", "three", "Both A and C"])
    assert not has_display_label_reference(["A,B", "A,C", "B,C", "A,B,C"])


def test_preprocessing_quarantines_unsafe_invalid_and_duplicate_rows_then_selects():
    rows = [
        _mmlu("safe 1", ["one", "two", "three", "four"]),
        _mmlu("relative", ["one", "two", "three", "none of the above"]),
        _mmlu("combination", ["one", "two", "three", "A and B"]),
        _mmlu("duplicate options", ["same", "same", "three", "four"]),
        _mmlu("invalid", ["one", "two"], 0),
        _mmlu("safe 2", ["five", "six", "seven", "eight"], 1),
    ]
    result = preprocess_rows(
        rows,
        adapter="mmlu",
        normalizer=normalize_row,
        policy=PreprocessingPolicy(),
    )
    assert [sample.question for sample in result.samples] == ["safe 1", "safe 2"]
    assert result.report["excluded_rows"] == 4
    assert result.report["excluded_by_reason"] == {
        "duplicate_option_text": 1,
        "normalization_error": 1,
        "option_label_combination": 1,
        "relative_position_reference": 1,
    }
    assert [sample.question for sample in result.select(1, 1).samples] == ["safe 2"]


def test_preprocessed_artifact_contains_four_prompts_per_sample_and_roundtrips(tmp_path):
    sample = MCQSample("q1", "mmlu", "Question", ("one", "two", "three", "four"), 2)
    rows = [_mmlu(sample.question, list(sample.options), sample.gold_index)]
    result = preprocess_rows(rows, adapter="mmlu", normalizer=normalize_row)
    prompt = MCQPrompt(PromptConfig("test-prompt-v1", "Choose exactly one."))
    artifact = write_preprocessed_artifact(
        result,
        tmp_path / "artifact",
        prompt_builder=prompt,
        dataset_metadata={"dataset_alias": "mmlu", "dataset_split": "test"},
    )
    loaded, manifest = load_preprocessed_artifact(artifact)
    assert loaded == result.samples
    assert manifest["schema_version"] == PREPROCESSED_SCHEMA_VERSION
    prompts = pd.read_parquet(artifact / "cyclic_prompts.parquet")
    assert len(prompts) == 4
    assert prompts["permutation_id"].tolist() == [0, 1, 2, 3]
    assert prompts["gold_displayed_label"].tolist() == ["C", "B", "A", "D"]


def test_artifact_checksum_detects_tampering(tmp_path):
    result = preprocess_rows(
        [_mmlu("Q", ["a", "b", "c", "d"])],
        adapter="mmlu",
        normalizer=normalize_row,
    )
    artifact = write_preprocessed_artifact(
        result,
        tmp_path / "artifact",
        prompt_builder=MCQPrompt(PromptConfig("v1", "Choose.")),
        dataset_metadata={"dataset_alias": "mmlu"},
    )
    with (artifact / "samples.parquet").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_preprocessed_artifact(artifact)


def test_policy_rejects_unknown_configuration_key():
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        PreprocessingPolicy.from_mapping({"typoed_filter": True})


def test_registry_applies_limit_after_filtering(monkeypatch):
    class FakeDataset(list):
        _fingerprint = "fake-fingerprint"

    dataset = FakeDataset(
        [
            _mmlu("excluded", ["one", "two", "three", "none of the above"]),
            _mmlu("accepted 1", ["a", "b", "c", "d"]),
            _mmlu("accepted 2", ["e", "f", "g", "h"]),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset),
    )
    samples, fingerprint, metadata = load_samples_with_metadata(
        {"alias": "mmlu", "repo": "fake", "config": "all", "adapter": "mmlu"},
        "test",
        limit=1,
        preprocessing_config=PreprocessingPolicy().to_dict(),
    )
    assert [sample.question for sample in samples] == ["accepted 1"]
    assert fingerprint == "fake-fingerprint"
    assert metadata["raw_rows"] == 3
    assert metadata["accepted_rows_before_selection"] == 2
    assert metadata["selected_samples"] == 1


def test_registry_rejects_artifact_for_wrong_dataset(tmp_path):
    result = preprocess_rows(
        [_mmlu("Q", ["a", "b", "c", "d"])],
        adapter="mmlu",
        normalizer=normalize_row,
    )
    artifact = write_preprocessed_artifact(
        result,
        tmp_path / "artifact",
        prompt_builder=MCQPrompt(PromptConfig("v1", "Choose.")),
        dataset_metadata={"dataset_alias": "mmlu", "dataset_split": "test"},
    )
    with pytest.raises(ValueError, match="artifact dataset"):
        load_samples_with_metadata(
            {"alias": "medqa", "repo": "fake", "adapter": "medqa"},
            "test",
            preprocessed=artifact,
        )

import pytest

from src.data.medmcqa import normalize_medmcqa
from src.data.medqa import normalize_medqa
from src.data.mmlu import normalize_mmlu


def test_medqa_joins_nonempty_question_fields() -> None:
    sample = normalize_medqa(
        {
            "id": "q1",
            "sent1": "Clinical stem",
            "sent2": "Follow-up question",
            "ending0": "one",
            "ending1": "two",
            "ending2": "three",
            "ending3": "four",
            "label": 2,
        },
        0,
    )
    assert sample.question == "Clinical stem\nFollow-up question"
    assert sample.options == ("one", "two", "three", "four")
    assert sample.gold_index == 2
    assert sample.gold_label == "C"


def test_medmcqa_normalization() -> None:
    sample = normalize_medmcqa(
        {
            "id": "m1",
            "question": "Question",
            "opa": "A content",
            "opb": "B content",
            "opc": "C content",
            "opd": "D content",
            "cop": 1,
            "subject_name": "Pathology",
        },
        0,
    )
    assert sample.dataset == "medmcqa"
    assert sample.subject == "Pathology"
    assert sample.gold_label == "B"


def test_mmlu_accepts_class_label_name() -> None:
    sample = normalize_mmlu(
        {
            "question": "Question",
            "choices": ["one", "two", "three", "four"],
            "answer": "D",
            "subject": "anatomy",
        },
        7,
    )
    assert sample.sample_id == "mmlu:anatomy:000007"
    assert sample.gold_index == 3


def test_adapter_rejects_non_four_option_mmlu() -> None:
    with pytest.raises(ValueError, match="four choices"):
        normalize_mmlu(
            {"question": "Q", "choices": ["a", "b"], "answer": 0, "subject": "x"},
            0,
        )


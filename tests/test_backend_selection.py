import pytest
import json
import subprocess
import sys

from src.models.token_scoring import FirstTokenOptionScorer, MockOptionScorer


def test_backend_labels_are_distinct():
    assert MockOptionScorer().backend == "mock"
    assert FirstTokenOptionScorer.backend == "real_transformers"


def test_real_scorer_requires_four_distinct_token_ids():
    with pytest.raises(ValueError):
        FirstTokenOptionScorer(object(), object(), [1, 1, 2, 3])  # not distinct
    with pytest.raises(ValueError):
        FirstTokenOptionScorer(object(), object(), [1, 2, 3])  # not four


def test_mock_scorer_is_deterministic_and_non_constant():
    mock = MockOptionScorer(seed=0)
    first = mock.score(["prompt-1", "prompt-2"])
    second = mock.score(["prompt-1", "prompt-2"])
    # deterministic within a process
    assert first[0].probabilities == second[0].probabilities
    # different prompts must not collapse to identical distributions
    assert first[0].probabilities != first[1].probabilities
    assert abs(sum(first[0].probabilities) - 1.0) < 1e-9


def test_mock_scorer_is_deterministic_across_python_processes():
    code = (
        "import json; from src.models.token_scoring import MockOptionScorer; "
        "print(json.dumps(MockOptionScorer(seed=42).score(['same prompt'])[0].probabilities))"
    )
    first = subprocess.check_output([sys.executable, "-c", code], text=True)
    second = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert json.loads(first) == json.loads(second)

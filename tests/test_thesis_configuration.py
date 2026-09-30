from src.utils.config import named_config


EXPECTED_MODELS = {
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


def test_exact_thesis_model_set_is_configured():
    experiment = named_config("experiments", "thesis")
    assert experiment["core_models"] == list(EXPECTED_MODELS)
    for alias, repo in EXPECTED_MODELS.items():
        config = named_config("models", alias)
        assert config["alias"] == alias
        assert config["model_repo"] == repo
        assert config["use_chat_template"] is True
        assert config["reasoning"] is False
        assert config["model_revision"] != "main"
        assert config["tokenizer_revision"] != "main"


def test_frozen_prompt_and_pride_design():
    experiment = named_config("experiments", "thesis")
    assert experiment["prompt"] == {
        "version": "thesis-mcq-v2",
        "instruction": (
            "Answer the following multiple-choice question.\n"
            "Respond with only the option letter (A, B, C, or D)."
        ),
        "question_heading": "Question:",
        "answer_heading": "Answer:",
    }
    assert experiment["pride"]["alpha"] == 0.05
    assert experiment["pride"]["alpha_rounding"] == "floor"
    assert experiment["pride"]["primary_seed"] == 42
    assert experiment["pride"]["sensitivity_seeds"] == [42, 43, 44, 45, 46]
    assert experiment["analysis"]["primary_pride_contrasts"] == [
        "in_domain_minus_default",
        "transfer_minus_default",
        "transfer_minus_in_domain",
    ]


def test_dataset_specific_default_splits():
    assert named_config("datasets", "medqa")["default_split"] == "test"
    assert named_config("datasets", "medmcqa")["default_split"] == "validation"
    assert named_config("datasets", "mmlu")["default_split"] == "test"

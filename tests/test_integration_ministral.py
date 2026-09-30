"""Real-model integration test for one configured thesis model.

Skipped by default. Run explicitly on adequate GPU hardware with:

    pytest -m integration

It will fail early (before any large download) via the capacity gate if the
current machine cannot hold the model.
"""
import pytest

pytestmark = pytest.mark.integration


def test_mistral_capacity_gate_and_load():
    from src.models.hardware import (
        InsufficientHardwareError,
        detect_hardware,
        format_report,
        require_capacity_for,
    )
    from src.utils.config import named_config

    config = named_config("models", "mistral-7b-instruct-v0.3")
    report = detect_hardware(config)
    print("\n" + format_report(report))

    # The capacity gate refuses inadequate hardware before any download. Here we
    # skip cleanly so `-m integration` is only meaningful on adequate GPU hardware.
    try:
        require_capacity_for(config, report)
    except InsufficientHardwareError as exc:
        pytest.skip(f"insufficient hardware for the 8B model: {exc}")

    # On adequate hardware, verify the real backend actually loads and scores.
    from src.models.loader import load_model
    from src.models.token_scoring import (
        FirstTokenOptionScorer,
        contextual_option_tokens,
        contextual_token_ids,
    )
    from src.prompts.mcq import MCQPrompt, PromptConfig
    from src.utils.config import load_yaml

    loaded = load_model(config)
    experiment = load_yaml("configs/experiments/thesis.yaml")
    prompt = MCQPrompt(PromptConfig.from_mapping(experiment["prompt"]))
    rendered = prompt.render(
        loaded.tokenizer,
        "What is 2 + 2?",
        ("3", "4", "5", "6"),
        bool(config.get("use_chat_template", True)),
        dict(config.get("chat_template_kwargs") or {}),
    )
    add_special_tokens = not bool(config.get("use_chat_template", True))
    rows = contextual_option_tokens(
        loaded.tokenizer, rendered, add_special_tokens=add_special_tokens
    )
    scorer = FirstTokenOptionScorer(
        loaded.model,
        loaded.tokenizer,
        contextual_token_ids(rows),
        add_special_tokens=add_special_tokens,
    )
    scores = scorer.score([rendered])
    assert abs(sum(scores[0].probabilities) - 1.0) < 1e-4

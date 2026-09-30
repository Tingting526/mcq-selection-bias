from __future__ import annotations

import argparse

from src.data.base import MCQSample
from src.models.loader import load_model
from src.models.token_scoring import contextual_option_tokens
from src.prompts.mcq import MCQPrompt, PromptConfig
from src.utils.config import load_yaml, named_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact A/B/C/D next-token representations")
    parser.add_argument(
        "--model",
        required=True,
        help="Model config alias, for example qwen2.5-7b-instruct",
    )
    parser.add_argument("--experiment", default="configs/experiments/thesis.yaml")
    args = parser.parse_args()

    model_config = named_config("models", args.model)
    experiment = load_yaml(args.experiment)
    loaded = load_model(model_config)
    prompt_builder = MCQPrompt(PromptConfig.from_mapping(experiment["prompt"]))
    sample = MCQSample(
        sample_id="diagnostic",
        dataset="diagnostic",
        question="Which option is the letter A?",
        options=("A", "B", "C", "D"),
        gold_index=0,
    )
    rendered = prompt_builder.render(
        loaded.tokenizer,
        sample.question,
        sample.options,
        bool(model_config.get("use_chat_template", True)),
        dict(model_config.get("chat_template_kwargs") or {}),
    )
    checks = contextual_option_tokens(
        loaded.tokenizer,
        rendered,
        add_special_tokens=not bool(model_config.get("use_chat_template", True)),
        raise_on_error=False,
    )
    print("Rendered prompt ending:")
    print(rendered[-800:])
    print("\nOption-token checks:")
    for check in checks:
        print(
            f"{check.label}: continuation={check.continuation_repr!r}, "
            f"token_id={check.token_id}, decoded={check.decoded_token!r}, "
            f"appended_count={check.appended_count}, "
            f"prefix_preserved={check.prefix_preserved}, valid={check.valid}"
        )
    if not all(check.valid for check in checks) or len({check.token_id for check in checks}) != 4:
        raise SystemExit("FAILED: A/B/C/D are not comparable single next tokens for this configuration.")
    print("\nPASS: A/B/C/D are distinct comparable single next tokens.")


if __name__ == "__main__":
    main()

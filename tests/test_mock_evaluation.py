from src.data.base import MCQSample
from src.evaluation.baseline import iter_baseline_batches
from src.models.token_scoring import OptionScore, check_option_tokens
from src.prompts.mcq import MCQPrompt, PromptConfig


class MockScorer:
    def score(self, prompts):
        return [OptionScore((1.0, 2.0, 0.0, -1.0), (0.2, 0.6, 0.15, 0.05)) for _ in prompts]


class CharacterTokenizer:
    def encode(self, value, add_special_tokens=False):
        return [ord(character) for character in value]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


def test_mock_scorer_runs_end_to_end_without_model_download() -> None:
    sample = MCQSample("q", "synthetic", "Question", ("one", "two", "three", "four"), 1)
    prompt = MCQPrompt(PromptConfig("test-v1", "Choose one."))
    batches = list(
        iter_baseline_batches(
            [sample],
            scorer=MockScorer(),
            prompt_builder=prompt,
            tokenizer=None,
            use_chat_template=False,
            model_name="mock",
            batch_size=1,
        )
    )
    assert batches[0][0]["predicted_label"] == "B"
    assert batches[0][0]["correct"] is True
    assert batches[0][0]["prob_B"] == 0.6


def test_frozen_thesis_prompt_is_rendered_exactly() -> None:
    prompt = MCQPrompt(
        PromptConfig(
            "thesis-mcq-v2",
            "Answer the following multiple-choice question.\n"
            "Respond with only the option letter (A, B, C, or D).",
        )
    )
    assert prompt.user_content("Which answer?", ("one", "two", "three", "four")) == (
        "Answer the following multiple-choice question.\n"
        "Respond with only the option letter (A, B, C, or D).\n\n"
        "Question: Which answer?\n\n"
        "A. one\n"
        "B. two\n"
        "C. three\n"
        "D. four\n\n"
        "Answer:"
    )


def test_option_token_check_uses_exact_prompt_boundary() -> None:
    checks = check_option_tokens(CharacterTokenizer(), "Answer:", "{label}")
    assert [check.token_ids for check in checks] == [(65,), (66,), (67,), (68,)]
    assert all(check.valid for check in checks)


def test_prompt_forwards_model_specific_chat_template_kwargs() -> None:
    class ChatTokenizer:
        chat_template = "configured"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "rendered"

    tokenizer = ChatTokenizer()
    prompt = MCQPrompt(PromptConfig("test-v1", "Choose one."))
    rendered = prompt.render(
        tokenizer,
        "Question",
        ("one", "two", "three", "four"),
        True,
        {"enable_thinking": False},
    )
    assert rendered == "rendered"
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.kwargs["tokenize"] is False
    assert tokenizer.kwargs["add_generation_prompt"] is True

"""Exercises the REAL FirstTokenOptionScorer code path (not the mock).

Uses a tiny deterministic torch module + tokenizer so the attention-mask
last-token indexing, softmax and finiteness checks run for real on CPU, with
no model download.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.token_scoring import FirstTokenOptionScorer, _last_nonpad_index


class TinyTokenizer:
    pad_token_id = 0

    def __init__(self, padding_side="left"):
        self.padding_side = padding_side
        self.last_add_special_tokens = None

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 60) + 1 for c in text]  # ids in 1..60 (0 reserved for pad)

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=True):
        self.last_add_special_tokens = add_special_tokens
        seqs = [([63] if add_special_tokens else []) + self.encode(t) for t in texts]
        max_len = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = [self.pad_token_id] * (max_len - len(s))
            if self.padding_side == "left":
                ids.append(pad + s)
                mask.append([0] * len(pad) + [1] * len(s))
            else:
                ids.append(s + pad)
                mask.append([1] * len(s) + [0] * len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class TinyModel:
    """Next-token logits depend deterministically on the token id at each position."""

    def __init__(self, vocab=64):
        self.vocab = vocab

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        base = torch.arange(self.vocab).float()
        ids = input_ids.float().unsqueeze(-1)  # (B, T, 1)
        logits = torch.sin((ids + 1) * 0.1 * base)  # (B, T, V)

        class Out:
            pass

        out = Out()
        out.logits = logits
        return out


def test_last_nonpad_index_left_and_right():
    left = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    assert _last_nonpad_index(left).tolist() == [3, 3]
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    assert _last_nonpad_index(right).tolist() == [1, 2]


@pytest.mark.parametrize("side", ["left", "right"])
def test_batch_matches_individual_regardless_of_padding(side):
    prompts = ["short one", "a considerably longer prompt than the first one here"]
    scorer_batched = FirstTokenOptionScorer(TinyModel(), TinyTokenizer(side), [1, 2, 3, 4])
    batched = scorer_batched.score(prompts)
    singles = [
        FirstTokenOptionScorer(TinyModel(), TinyTokenizer(side), [1, 2, 3, 4]).score([p])[0]
        for p in prompts
    ]
    for b, s in zip(batched, singles):
        assert np.allclose(b.probabilities, s.probabilities, atol=1e-5)
        assert np.allclose(b.logits, s.logits, atol=1e-5)


def test_probabilities_are_normalized_and_finite():
    scorer = FirstTokenOptionScorer(TinyModel(), TinyTokenizer("left"), [1, 2, 3, 4])
    for score in scorer.score(["abc", "defgh"]):
        assert abs(sum(score.probabilities) - 1.0) < 1e-5
        assert all(np.isfinite(score.probabilities))
        assert all(np.isfinite(score.logits))


def test_scorer_backend_label_is_real():
    assert FirstTokenOptionScorer.backend == "real_transformers"


def test_chat_template_text_is_tokenized_without_duplicate_special_tokens():
    tokenizer = TinyTokenizer("left")
    scorer = FirstTokenOptionScorer(TinyModel(), tokenizer, [1, 2, 3, 4])
    scorer.score(["rendered chat prompt"])
    assert tokenizer.last_add_special_tokens is False

import pytest

from src.models.token_scoring import contextual_option_tokens, contextual_token_ids


class CharTokenizer:
    """Character-level tokenizer: every character is exactly one token."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class TwoTokenLabelTokenizer:
    """A tokenizer where a bare label letter costs TWO tokens (never single-token)."""

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        if text and text[-1] in "ABCD":
            ids.append(ord(text[-1]))  # a second token for the trailing label
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class SpacePreferringTokenizer:
    """Bare 'A' costs two tokens, but ' A' is a single merged token (like BPE 'ĠA')."""

    SPACE_LABEL = {f" {label}": 2000 + ord(label) for label in "ABCD"}

    def encode(self, text, add_special_tokens=False):
        ids: list[int] = []
        i = 0
        while i < len(text):
            two = text[i : i + 2]
            if two in self.SPACE_LABEL:
                ids.append(self.SPACE_LABEL[two])
                i += 2
            elif text[i] in "ABCD":
                ids.extend([ord(text[i]), ord(text[i])])  # bare label -> two tokens
                i += 1
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        reverse = {v: k for k, v in self.SPACE_LABEL.items()}
        return "".join(reverse[i] if i in reverse else chr(i) for i in ids)


def test_contextual_extraction_single_token_no_space():
    rows = contextual_option_tokens(CharTokenizer(), "Question\nAnswer:")
    assert [r.continuation_repr for r in rows] == ["A", "B", "C", "D"]
    assert contextual_token_ids(rows) == [ord("A"), ord("B"), ord("C"), ord("D")]
    assert all(r.appended_count == 1 and r.prefix_preserved for r in rows)


def test_contextual_falls_back_to_leading_space_representation():
    # prompt deliberately avoids uppercase A-D so it is not corrupted by the tokenizer
    rows = contextual_option_tokens(SpacePreferringTokenizer(), "Question here\nreply:")
    assert [r.continuation_repr for r in rows] == [" A", " B", " C", " D"]
    assert contextual_token_ids(rows) == [2000 + ord(c) for c in "ABCD"]


def test_contextual_fails_loudly_when_label_is_multi_token():
    with pytest.raises(ValueError):
        contextual_option_tokens(TwoTokenLabelTokenizer(), "prompt\nAnswer:")


def test_contextual_no_raise_returns_invalid_rows():
    rows = contextual_option_tokens(
        TwoTokenLabelTokenizer(), "prompt\nAnswer:", raise_on_error=False
    )
    assert not all(r.valid for r in rows)
    with pytest.raises(ValueError):
        contextual_token_ids(rows)

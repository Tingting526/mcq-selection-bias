from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from src import OPTION_LABELS


@dataclass(frozen=True, slots=True)
class OptionScore:
    logits: tuple[float, float, float, float]
    probabilities: tuple[float, float, float, float]

    @property
    def predicted_index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def predicted_label(self) -> str:
        return OPTION_LABELS[self.predicted_index]


class OptionScorer(Protocol):
    backend: str

    def score(self, prompts: Sequence[str]) -> list[OptionScore]: ...


# ===========================================================================
# Isolated (legacy) option-token check
# ===========================================================================
@dataclass(frozen=True, slots=True)
class OptionTokenCheck:
    label: str
    candidate_text: str
    token_ids: tuple[int, ...]
    decoded_tokens: tuple[str, ...]
    exactly_one_token: bool
    decoded_as_label: bool
    boundary_stable: bool

    @property
    def valid(self) -> bool:
        return self.exactly_one_token and self.decoded_as_label and self.boundary_stable


def check_option_tokens(
    tokenizer: Any,
    rendered_prompt: str,
    token_template: str,
    *,
    raise_on_error: bool = True,
) -> list[OptionTokenCheck]:
    """Legacy isolated check. Kept for CPU tests and backward compatibility.

    Prefer :func:`contextual_option_tokens` for the real methodological check,
    which appends each candidate to the EXACT rendered prompt.
    """
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    checks = []
    for label in OPTION_LABELS:
        candidate_text = token_template.format(label=label)
        token_ids = tuple(tokenizer.encode(candidate_text, add_special_tokens=False))
        decoded_tokens = tuple(
            tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids
        )
        decoded_as_label = len(token_ids) == 1 and decoded_tokens[0].strip() == label
        if len(token_ids) == 1:
            extended = tokenizer.decode(prompt_ids + [token_ids[0]], skip_special_tokens=False)
            boundary_stable = (
                extended.startswith(decoded_prompt)
                and extended[len(decoded_prompt) :].strip() == label
            )
        else:
            boundary_stable = False
        checks.append(
            OptionTokenCheck(
                label=label,
                candidate_text=candidate_text,
                token_ids=token_ids,
                decoded_tokens=decoded_tokens,
                exactly_one_token=len(token_ids) == 1,
                decoded_as_label=decoded_as_label,
                boundary_stable=boundary_stable,
            )
        )
    ids = [check.token_ids[0] for check in checks if check.exactly_one_token]
    if raise_on_error and (not all(check.valid for check in checks) or len(set(ids)) != 4):
        details = "; ".join(
            f"{check.label}={list(check.token_ids)} decoded={list(check.decoded_tokens)!r} "
            f"single={check.exactly_one_token} boundary={check.boundary_stable}"
            for check in checks
        )
        raise ValueError(
            "Option-token safety check failed under the exact rendered prompt. "
            "A/B/C/D must be distinct, comparable single next tokens. " + details
        )
    return checks


# ===========================================================================
# Contextual option-token diagnostic (the methodologically correct one)
# ===========================================================================
@dataclass(frozen=True, slots=True)
class ContextualOptionToken:
    label: str
    continuation_repr: str          # e.g. "A" or " A" — what actually got appended
    token_id: int
    decoded_token: str
    appended_count: int             # tokens added beyond the prompt (must be 1)
    prefix_preserved: bool          # prompt tokenization unchanged by the continuation

    @property
    def valid(self) -> bool:
        return self.appended_count == 1 and self.prefix_preserved


def contextual_option_tokens(
    tokenizer: Any,
    rendered_prompt: str,
    *,
    continuation_templates: Sequence[str] = ("{label}", " {label}"),
    add_special_tokens: bool = False,
    raise_on_error: bool = True,
) -> list[ContextualOptionToken]:
    """Determine the single next-token id for each of A/B/C/D under the EXACT prompt.

    For each candidate whitespace representation (e.g. ``"A"`` vs ``" A"``) it checks
    that appending the continuation to the rendered prompt:
      * leaves the prompt's own token sequence unchanged (prefix preserved), and
      * adds exactly ONE token.
    A single representation must work for all four labels and yield four distinct
    token ids, otherwise this fails loudly (unless ``raise_on_error=False``).
    """
    base_ids = list(tokenizer.encode(rendered_prompt, add_special_tokens=add_special_tokens))
    n = len(base_ids)

    def evaluate(template: str) -> list[ContextualOptionToken]:
        rows: list[ContextualOptionToken] = []
        for label in OPTION_LABELS:
            cont = template.format(label=label)
            ext = list(tokenizer.encode(rendered_prompt + cont, add_special_tokens=add_special_tokens))
            prefix_ok = ext[:n] == base_ids
            appended = ext[n:] if prefix_ok else ext[len(base_ids):]
            token_id = int(appended[0]) if appended else -1
            decoded = tokenizer.decode([token_id], skip_special_tokens=False) if token_id >= 0 else ""
            rows.append(
                ContextualOptionToken(
                    label=label,
                    continuation_repr=cont,
                    token_id=token_id,
                    decoded_token=decoded,
                    appended_count=len(ext) - n if prefix_ok else len(appended),
                    prefix_preserved=prefix_ok,
                )
            )
        return rows

    chosen: list[ContextualOptionToken] | None = None
    last_attempt: list[ContextualOptionToken] = []
    for template in continuation_templates:
        rows = evaluate(template)
        last_attempt = rows
        ids = [row.token_id for row in rows]
        if all(row.valid for row in rows) and len(set(ids)) == 4:
            chosen = rows
            break

    result = chosen if chosen is not None else last_attempt
    if raise_on_error and chosen is None:
        details = "; ".join(
            f"{row.label}: repr={row.continuation_repr!r} appended={row.appended_count} "
            f"prefix_preserved={row.prefix_preserved} token_id={row.token_id} decoded={row.decoded_token!r}"
            for row in result
        )
        raise ValueError(
            "Contextual option-token diagnostic FAILED: A/B/C/D are not all distinct "
            "single-token continuations of the exact rendered prompt. Refusing to score. "
            + details
        )
    return result


def contextual_token_ids(rows: Sequence[ContextualOptionToken]) -> list[int]:
    if len(rows) != 4 or not all(row.valid for row in rows):
        raise ValueError("A successful four-label contextual diagnostic is required before scoring")
    return [row.token_id for row in rows]


def diagnostic_to_dict(rows: Sequence[ContextualOptionToken]) -> dict[str, Any]:
    return {
        "labels": list(OPTION_LABELS),
        "candidates": [
            {
                "label": row.label,
                "continuation_repr": row.continuation_repr,
                "token_id": row.token_id,
                "decoded_token": row.decoded_token,
                "appended_count": row.appended_count,
                "prefix_preserved": row.prefix_preserved,
                "valid": row.valid,
            }
            for row in rows
        ],
        "token_ids": [row.token_id for row in rows],
        "all_valid": all(row.valid for row in rows),
        "distinct_ids": len({row.token_id for row in rows}) == 4,
    }


# ===========================================================================
# Real Transformers first-token scorer
# ===========================================================================
def _last_nonpad_index(attention_mask: Any) -> Any:
    """Index of the last real (non-pad) token per row. Works for left OR right padding."""
    import torch

    seq_len = attention_mask.shape[1]
    # first 1 scanning from the end == last 1 in the original row
    first_one_from_end = torch.flip(attention_mask, dims=[1]).to(torch.int64).argmax(dim=1)
    return seq_len - 1 - first_one_from_end


class FirstTokenOptionScorer:
    """Real first-token option-ID scorer over a Hugging Face causal/CLM-style model."""

    backend = "real_transformers"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        token_ids: Sequence[int],
        *,
        add_special_tokens: bool = False,
    ):
        ids = [int(t) for t in token_ids]
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("Exactly four distinct option-token ids are required before scoring")
        self.model = model
        self.tokenizer = tokenizer
        self.token_ids = ids
        self.add_special_tokens = bool(add_special_tokens)

    @classmethod
    def from_contextual(
        cls, model: Any, tokenizer: Any, rows: Sequence[ContextualOptionToken]
    ) -> "FirstTokenOptionScorer":
        return cls(model, tokenizer, contextual_token_ids(rows))

    def score(self, prompts: Sequence[str]) -> list[OptionScore]:
        if not prompts:
            return []
        import torch

        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=self.add_special_tokens,
        )
        device = getattr(self.model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            output = self.model(**encoded)

        logits = output.logits  # (batch, seq, vocab)
        attention_mask = encoded["attention_mask"]
        last_index = _last_nonpad_index(attention_mask)  # (batch,)
        batch_index = torch.arange(logits.shape[0], device=logits.device)
        next_token_logits = logits[batch_index, last_index, :]  # (batch, vocab)

        option_logits = next_token_logits[:, self.token_ids].float()
        probabilities = torch.softmax(option_logits, dim=-1)

        logits_cpu = option_logits.detach().cpu().numpy()
        probabilities_cpu = probabilities.detach().cpu().numpy()

        if not np.all(np.isfinite(logits_cpu)):
            raise ValueError("Non-finite option logits produced by the model forward pass")
        row_sums = probabilities_cpu.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            raise ValueError(f"Option probabilities do not sum to 1: {row_sums.tolist()}")

        return [
            OptionScore(
                logits=tuple(float(v) for v in logits_row),  # type: ignore[arg-type]
                probabilities=tuple(float(v) for v in probability_row),  # type: ignore[arg-type]
            )
            for logits_row, probability_row in zip(logits_cpu, probabilities_cpu, strict=True)
        ]


# ===========================================================================
# Deterministic mock scorer (explicit opt-in only; never a silent fallback)
# ===========================================================================
class MockOptionScorer:
    """Deterministic, model-free scorer for CPU pipeline testing (``--scorer mock``)."""

    backend = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, prompts: Sequence[str]) -> list[OptionScore]:
        out: list[OptionScore] = []
        for prompt in prompts:
            digest = hashlib.sha256(f"{self.seed}\0{prompt}".encode("utf-8")).digest()
            stable_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
            rng = np.random.default_rng(stable_seed)
            logits = rng.normal(size=4)
            shifted = logits - logits.max()
            probs = np.exp(shifted)
            probs = probs / probs.sum()
            out.append(
                OptionScore(
                    logits=tuple(float(v) for v in logits),  # type: ignore[arg-type]
                    probabilities=tuple(float(v) for v in probs),  # type: ignore[arg-type]
                )
            )
        return out

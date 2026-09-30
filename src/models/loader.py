from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .hardware import HardwareReport, require_capacity_for, resolve_dtype_name


@dataclass(slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    config: dict[str, Any]
    resolved_model_revision: str | None
    resolved_tokenizer_revision: str | None
    model_class_name: str
    tokenizer_class_name: str
    hardware: HardwareReport | None = None


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose bfloat16, float16, or float32") from exc


def _resolve_model_class(name: str) -> Any:
    import transformers

    try:
        return getattr(transformers, name)
    except AttributeError as exc:
        raise ValueError(
            f"transformers has no model class {name!r}. Installed transformers "
            f"{getattr(transformers, '__version__', '?')} may be too old for this checkpoint."
        ) from exc


def _load_tokenizer(config: Mapping[str, Any]) -> Any:
    """Load a tokenizer for one of the configured Transformers checkpoints."""
    repo = str(config.get("tokenizer_repo") or config["model_repo"])
    revision = config.get("tokenizer_revision")
    trust = bool(config.get("trust_remote_code", False))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        repo,
        revision=revision,
        trust_remote_code=trust,
    )

    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token for batching")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: Mapping[str, Any], *, enforce_capacity: bool = True) -> LoadedModel:
    """Load a real model + tokenizer. Fails EARLY (before download) on insufficient hardware."""
    # --- capacity gate FIRST, so we never half-download a model that cannot load ---
    report: HardwareReport | None = None
    if enforce_capacity:
        report = require_capacity_for(config)

    tokenizer = _load_tokenizer(config)

    model_class_name = str(config.get("model_class", "AutoModelForCausalLM"))
    model_class = _resolve_model_class(model_class_name)

    dtype_name = resolve_dtype_name(
        str(config.get("dtype", "auto")),
        report.selected_device if report is not None else "cpu",
    )
    torch_dtype: Any = "auto" if str(config.get("dtype", "auto")) == "auto" else _torch_dtype(dtype_name)

    kwargs: dict[str, Any] = {
        "revision": config.get("model_revision"),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "torch_dtype": torch_dtype,
    }
    if config.get("device_map") is not None:
        kwargs["device_map"] = config["device_map"]
    if config.get("attn_implementation"):
        kwargs["attn_implementation"] = config["attn_implementation"]

    model = model_class.from_pretrained(str(config["model_repo"]), **kwargs)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=dict(config),
        resolved_model_revision=getattr(model.config, "_commit_hash", None),
        resolved_tokenizer_revision=getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        model_class_name=model_class_name,
        tokenizer_class_name=type(tokenizer).__name__,
        hardware=report,
    )

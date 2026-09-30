"""Hardware / environment detection and an early-fail capacity gate.

The thesis model (Ministral-3-8B) must never be silently loaded onto CPU or
half-downloaded onto a machine that cannot hold it. This module reports the
environment and refuses, *before* any large download, when the selected
accelerator cannot realistically fit the configured model.
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class HardwareReport:
    platform: str
    python_version: str
    torch_version: str | None
    transformers_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_memory_gb: float | None
    mps_available: bool
    system_ram_gb: float | None
    selected_device: str
    selected_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _system_ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def _package_version(name: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_dtype_name(requested: str, device: str) -> str:
    """Map a configured dtype ('auto' or explicit) to a concrete dtype name."""
    requested = str(requested).lower()
    if requested != "auto":
        return requested
    return {"cuda": "bfloat16", "mps": "float16", "cpu": "float32"}[device]


def _preferred_device(prefer_cpu: bool = False) -> str:
    if prefer_cpu:
        return "cpu"
    try:
        import torch
    except Exception:  # torch not installed: only mock scoring is possible
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _accelerator_memory_gb(device: str) -> float | None:
    """Memory the *selected accelerator* can realistically use, in GB."""
    try:
        import torch
    except Exception:
        return _system_ram_gb()
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    if device == "mps":
        # Apple unified memory: prefer torch's recommended working-set ceiling,
        # otherwise fall back to total system RAM.
        recommended = getattr(getattr(torch, "mps", None), "recommended_max_memory", None)
        if callable(recommended):
            try:
                value = recommended()
                if value:
                    return value / 1e9
            except Exception:
                pass
        return _system_ram_gb()
    return _system_ram_gb()


def detect_hardware(model_config: Mapping[str, Any] | None = None) -> HardwareReport:
    torch_version = _package_version("torch")
    cuda_available = False
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    mps_available = False

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_memory_gb = props.total_memory / 1e9
        mps_available = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
    except Exception:
        pass

    prefer_cpu = bool(model_config.get("device_map") == "cpu") if model_config else False
    device = _preferred_device(prefer_cpu=prefer_cpu)
    requested_dtype = str(model_config.get("dtype", "auto")) if model_config else "auto"
    dtype = resolve_dtype_name(requested_dtype, device)

    return HardwareReport(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        torch_version=torch_version,
        transformers_version=_package_version("transformers"),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        mps_available=mps_available,
        system_ram_gb=_system_ram_gb(),
        selected_device=device,
        selected_dtype=dtype,
    )


_BYTES_PER_PARAM = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0, "fp8": 1.0, "float8": 1.0}


def estimate_weight_memory_gb(params_billions: float, dtype: str) -> float:
    """Rough weight-only memory footprint. Real usage adds activations/KV cache."""
    bytes_per = _BYTES_PER_PARAM.get(str(dtype).lower(), 2.0)
    return params_billions * 1e9 * bytes_per / 1e9


def format_report(report: HardwareReport) -> str:
    lines = [
        "Environment / hardware report",
        "-----------------------------",
        f"platform             : {report.platform}",
        f"python               : {report.python_version}",
        f"torch                : {report.torch_version}",
        f"transformers         : {report.transformers_version}",
        f"CUDA available       : {report.cuda_available}",
        f"CUDA version         : {report.cuda_version}",
        f"GPU name             : {report.gpu_name}",
        f"GPU VRAM (GB)        : {report.gpu_memory_gb if report.gpu_memory_gb is None else round(report.gpu_memory_gb, 1)}",
        f"MPS available        : {report.mps_available}",
        f"system RAM (GB)      : {report.system_ram_gb if report.system_ram_gb is None else round(report.system_ram_gb, 1)}",
        f"selected device      : {report.selected_device}",
        f"selected dtype       : {report.selected_dtype}",
    ]
    return "\n".join(lines)


class InsufficientHardwareError(RuntimeError):
    """Raised before any large download when the machine cannot fit the model."""


def require_capacity_for(
    model_config: Mapping[str, Any],
    report: HardwareReport | None = None,
    *,
    headroom_factor: float = 1.3,
) -> HardwareReport:
    """Fail early (before download) unless the model can realistically be loaded.

    Rules:
    - A model flagged ``requires_accelerator: true`` may not run on CPU.
    - Estimated weight memory * ``headroom_factor`` must fit in the selected
      accelerator's memory (VRAM for CUDA, unified RAM for MPS).
    """
    report = report or detect_hardware(model_config)
    name = model_config.get("name") or model_config.get("alias") or model_config.get("model_repo")

    requires_accel = bool(model_config.get("requires_accelerator", True))
    if requires_accel and report.selected_device == "cpu":
        raise InsufficientHardwareError(
            f"Model '{name}' requires a GPU/accelerator but only CPU is available. "
            "This thesis model must not run on CPU. Use --scorer mock for CPU testing, "
            "or run on a CUDA GPU (see check_hardware output for the required VRAM)."
        )

    params_b = model_config.get("approx_params_billions")
    if params_b is None:
        return report  # cannot estimate; leave it to the loader

    needed = estimate_weight_memory_gb(float(params_b), report.selected_dtype)
    min_required = model_config.get("min_accelerator_memory_gb")
    required_gb = float(min_required) if min_required is not None else needed * headroom_factor
    # CUDA => dedicated VRAM; MPS/CPU => unified/system RAM (both taken from the report).
    available = report.gpu_memory_gb if report.selected_device == "cuda" else report.system_ram_gb

    if available is not None and available < required_gb:
        raise InsufficientHardwareError(
            f"Model '{name}' needs about {required_gb:.0f} GB on the '{report.selected_device}' "
            f"device (~{needed:.0f} GB weights in {report.selected_dtype} x {headroom_factor} headroom), "
            f"but only {available:.0f} GB is available. Refusing to download/load the model.\n"
            "Options: (1) run on a GPU with enough VRAM, e.g. one 24 GB card (RTX 4090/L4/A10) "
            "for bf16, or a 16 GB card only with an approved quantized checkpoint; "
            "(2) use --scorer mock for CPU-only pipeline testing."
        )
    return report

import pytest

from src.models.hardware import (
    HardwareReport,
    InsufficientHardwareError,
    estimate_weight_memory_gb,
    require_capacity_for,
    resolve_dtype_name,
)


def make_report(device, *, gpu_gb=None, ram_gb=None, dtype="bfloat16"):
    return HardwareReport(
        platform="test",
        python_version="3.12",
        torch_version="x",
        transformers_version="y",
        cuda_available=device == "cuda",
        cuda_version=None,
        gpu_name=None,
        gpu_memory_gb=gpu_gb,
        mps_available=device == "mps",
        system_ram_gb=ram_gb,
        selected_device=device,
        selected_dtype=dtype,
    )


CONFIG = {
    "name": "qwen3-8b",
    "approx_params_billions": 8.0,
    "min_accelerator_memory_gb": 22,
    "requires_accelerator": True,
}


def test_cpu_is_refused_for_accelerator_model():
    with pytest.raises(InsufficientHardwareError):
        require_capacity_for(CONFIG, make_report("cpu", ram_gb=64))


def test_insufficient_vram_is_refused():
    with pytest.raises(InsufficientHardwareError):
        require_capacity_for(CONFIG, make_report("cuda", gpu_gb=8))


def test_mps_unified_memory_too_small_is_refused():
    # e.g. an Apple machine with 17 GB unified memory
    with pytest.raises(InsufficientHardwareError):
        require_capacity_for(CONFIG, make_report("mps", ram_gb=17))


def test_sufficient_vram_passes():
    report = require_capacity_for(CONFIG, make_report("cuda", gpu_gb=40))
    assert report.selected_device == "cuda"


def test_resolve_dtype_auto_per_device():
    assert resolve_dtype_name("auto", "cuda") == "bfloat16"
    assert resolve_dtype_name("auto", "mps") == "float16"
    assert resolve_dtype_name("auto", "cpu") == "float32"
    assert resolve_dtype_name("float16", "cuda") == "float16"


def test_weight_memory_estimate():
    assert estimate_weight_memory_gb(8.0, "bfloat16") == pytest.approx(16.0, rel=0.01)
    assert estimate_weight_memory_gb(8.0, "float32") == pytest.approx(32.0, rel=0.01)

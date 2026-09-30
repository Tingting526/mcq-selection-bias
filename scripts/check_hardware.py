from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT  # noqa: F401
from src.models.hardware import (
    InsufficientHardwareError,
    detect_hardware,
    estimate_weight_memory_gb,
    format_report,
    require_capacity_for,
)
from src.utils.config import named_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the environment and check whether a model can be loaded here"
    )
    parser.add_argument("--model", help="Model config alias to run the capacity gate against")
    args = parser.parse_args()

    model_config = named_config("models", args.model) if args.model else None
    report = detect_hardware(model_config)
    print(format_report(report))

    if model_config is None:
        print("\n(no --model given; skipped the capacity gate)")
        return

    params_b = model_config.get("approx_params_billions")
    if params_b is not None:
        est = estimate_weight_memory_gb(float(params_b), report.selected_dtype)
        print(
            f"\nModel '{args.model}': ~{params_b}B params "
            f"=> ~{est:.0f} GB weights in {report.selected_dtype}"
        )
    try:
        require_capacity_for(model_config, report)
    except InsufficientHardwareError as exc:
        print("\nCAPACITY GATE: FAIL")
        print(str(exc))
        raise SystemExit(2)
    print("\nCAPACITY GATE: PASS - this machine can attempt to load the model.")


if __name__ == "__main__":
    main()

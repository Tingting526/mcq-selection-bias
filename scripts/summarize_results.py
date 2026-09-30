from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT
from src.metrics.selection_bias import permutation_accuracy_metrics, selection_bias_metrics
from src.utils.io import unique_path


def _read_input(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    parts = sorted((path / "parts").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet result partitions found under {path}")
    return pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize raw baseline, permutation, or PriDe records")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--by-subject", action="store_true")
    parser.add_argument("--prediction-column", default="predicted_label")
    args = parser.parse_args()

    frame = _read_input(args.input)
    groups = [("pooled", frame)]
    if args.by_subject and "subject" in frame and frame["subject"].notna().any():
        groups.extend((str(subject), subset) for subject, subset in frame.groupby("subject"))
    summaries = []
    for subject, subset in groups:
        rows = subset.to_dict(orient="records")
        metrics = selection_bias_metrics(rows, prediction_key=args.prediction_column)
        if "permutation_id" in subset and set(subset["permutation_id"].astype(int)) == {0, 1, 2, 3}:
            metrics.update(permutation_accuracy_metrics(rows))
        summaries.append({"subject": subject, **metrics})
    output = unique_path(PROJECT_ROOT / "outputs" / "summaries", "summary", ".csv")
    pd.DataFrame(summaries).to_csv(output, index=False)
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()


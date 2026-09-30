"""Join matched pipeline summaries without mixing mock, smoke and full runs."""
import argparse
from pathlib import Path
import pandas as pd
from _bootstrap import PROJECT_ROOT


def aggregate(directory, pipeline_id, scorer):
    frames = []
    backend = 'mock' if scorer == 'mock' else 'real_transformers'
    for path in sorted(Path(directory).glob('*.csv'), key=lambda p: p.stat().st_mtime_ns):
        frame = pd.read_csv(path)
        if {'pipeline_id', 'kind', 'scorer_backend', 'run_dir'}.issubset(frame.columns):
            frames.append(frame[(frame.pipeline_id == pipeline_id) & (frame.scorer_backend == backend)])
    if not frames or not any(len(f) for f in frames):
        raise ValueError('No matching managed summaries')
    data = pd.concat(frames, ignore_index=True).drop_duplicates('run_dir', keep='last')
    keys = ['pipeline_id', 'model_repo', 'dataset_repo', 'scorer_backend', 'seed', 'sample_set_sha256']
    rows = []
    for key, group in data.groupby(keys, dropna=False):
        if set(group.kind) != {'baseline', 'permutations', 'pride'} or len(group) != 3:
            raise ValueError('Incomplete or ambiguous pipeline summaries')
        row = dict(zip(keys, key))
        for _, item in group.iterrows():
            for name, value in item.items():
                if name not in keys and name != 'kind' and (pd.notna(value) or 'rstd' in name):
                    row[f'{item["kind"]}_{name}'] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=PROJECT_ROOT / 'outputs/summaries')
    p.add_argument('--pipeline-id', required=True, action='append')
    p.add_argument('--scorer', choices=['mock', 'real'], default='real')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = pd.concat([aggregate(args.input, item, args.scorer) for item in args.pipeline_id], ignore_index=True)
    if result.duplicated(["model_repo", "dataset_repo"]).any():
        raise ValueError("Select exactly one pipeline per model/dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Comparison: {args.output}')

if __name__ == '__main__':
    main()

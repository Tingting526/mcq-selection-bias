"""Plan, submit and collect explicit model x dataset Slurm experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

import pandas as pd
from _bootstrap import PROJECT_ROOT
from aggregate_results import aggregate
from run_pipeline import artifact_for
from src.utils.config import named_config
from src.utils.io import write_json


def plan(directory, models, datasets, limit=None, batch_size=1, preprocessed=None):
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    directory = Path(directory).resolve()
    if preprocessed is not None and len(set(datasets)) != 1:
        raise ValueError("Explicit preprocessed artifact requires one dataset")
    entries = []
    for model in dict.fromkeys(models):
        named_config('models', model)  # Fail before creating a partial plan.
        for dataset in dict.fromkeys(datasets):
            config = dict(model=model, dataset=dataset, scorer='real', limit=limit,
                          batch_size=batch_size, preprocessed=str(Path(preprocessed or artifact_for(dataset)).resolve()))
            entries.append(dict(config=config, pipeline=str(directory / f'{model}_{dataset}_{uuid.uuid4().hex[:12]}'),
                                job_id=None))
    directory.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        write_json(Path(entry['pipeline']) / 'pipeline.json',
                   dict(config=entry['config'], steps={}))
    state = dict(entries=entries)
    write_json(directory / 'campaign.json', state)
    return state


def job_environment(entry):
    env = os.environ.copy()
    # Explicitly replace inherited settings from previous interactive test runs.
    for name in ['LIMIT', 'RESUME', 'PRIOR', 'PREPROCESSED']:
        env.pop(name, None)
    config = entry['config']
    env.update(MODEL=config['model'], DATASET=config['dataset'], STEP='pipeline',
               BATCH_SIZE=str(config['batch_size']), PREPROCESSED=config['preprocessed'],
               RESUME=entry['pipeline'])
    if config['limit'] is not None:
        env['LIMIT'] = str(config['limit'])
    return env


def submit(directory, state, partition):
    for entry in state['entries']:
        if entry['job_id']:
            print(f"Already submitted: {entry['job_id']} {entry['pipeline']}")
            continue
        if entry.get('submission_started'):
            raise ValueError('Previous submission outcome uncertain; inspect squeue/sacct before editing campaign.json')
        entry['submission_started'] = True
        write_json(directory / 'campaign.json', state, overwrite=True)
        env = job_environment(entry)
        env['PARTITION'] = partition
        result = subprocess.run(['bash', 'scripts/slurm/submit.sh', 'analysis'],
                                cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
        if result.returncode:
            entry['submission_started'] = False
            write_json(directory / 'campaign.json', state, overwrite=True)
            raise RuntimeError(result.stderr or result.stdout)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise ValueError(f'Unexpected submission output; inspect Slurm before retry: {result.stdout}')
        entry.update(job_id=job, partition=partition, submission_started=False)
        write_json(directory / 'campaign.json', state, overwrite=True)
        print(f"Submitted {job}: {entry['config']['model']} / {entry['config']['dataset']}")


def collect(directory, state):
    frames = []
    for entry in state['entries']:
        pipeline = Path(entry['pipeline'])
        status = json.loads((pipeline / 'pipeline.json').read_text())
        if status.get('status') != 'completed':
            raise ValueError(f'Pipeline not completed: {pipeline}')
        frames.append(aggregate(PROJECT_ROOT / 'outputs/summaries', pipeline.name, 'real'))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(directory / 'comparison.csv', index=False)
    print(f"Comparison: {directory / 'comparison.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'submit', 'collect'])
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--models', nargs='+')
    p.add_argument('--datasets', nargs='+', choices=['medqa', 'medmcqa', 'mmlu'],
                   default=['medqa', 'medmcqa', 'mmlu'])
    p.add_argument('--limit', type=int)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--partition')
    p.add_argument('--preprocessed', type=Path)
    args = p.parse_args()
    directory = args.directory.resolve()
    if args.action == 'plan':
        models = args.models or named_config('experiments', 'thesis')['core_models']
        state = plan(directory, models, args.datasets, args.limit, args.batch_size, args.preprocessed)
        print(f"Prepared {len(state['entries'])} jobs; limit={args.limit or 'FULL'}. Nothing submitted.")
    else:
        state = json.loads((directory / 'campaign.json').read_text())
        if args.action == 'submit':
            if not args.partition:
                p.error('--partition is required for submission')
            submit(directory, state, args.partition)
        else:
            collect(directory, state)


if __name__ == '__main__':
    main()

"""One model/dataset pipeline; mock on CPU, real only inside a GPU allocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from _bootstrap import PROJECT_ROOT
from src.utils.io import write_json


def artifact_for(dataset: str) -> Path:
    root = PROJECT_ROOT / 'outputs' / 'preprocessed' / dataset
    candidates = list(root.rglob('manifest.json'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one artifact under {root}; select explicitly with --preprocessed')
    return candidates[0].parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen2.5-7b-instruct')
    parser.add_argument('--dataset', choices=['medqa', 'medmcqa', 'mmlu'], default='medqa')
    parser.add_argument('--scorer', choices=['real', 'mock'], default='real')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--preprocessed', type=Path)
    parser.add_argument('--resume', type=Path, help='Pipeline directory, not an individual raw run')
    args = parser.parse_args()
    if args.scorer == 'real' and not os.environ.get('SLURM_JOB_ID'):
        parser.error('Real pipelines require a Slurm GPU allocation')
    config = dict(model=args.model, dataset=args.dataset, scorer=args.scorer,
                  limit=args.limit, batch_size=args.batch_size,
                  preprocessed=str((args.preprocessed or artifact_for(args.dataset)).resolve()))
    directory = args.resume or PROJECT_ROOT / 'outputs' / 'pipelines' / f'{args.scorer}_{args.model}_{args.dataset}_{uuid.uuid4().hex[:12]}'
    directory = directory.resolve()
    state_path = directory / 'pipeline.json'
    if args.resume:
        state = json.loads(state_path.read_text())
        if state['config'] != config:
            raise ValueError('Resume configuration differs from pipeline.json')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        state = {'config': config, 'steps': {}}
        write_json(state_path, state)
    state['status'] = 'running'
    write_json(state_path, state, overwrite=True)
    os.environ['MCQ_PIPELINE_ID'] = directory.name
    print(f'Pipeline: {directory}', flush=True)
    common = ['--model', args.model, '--dataset', args.dataset, '--scorer', args.scorer,
              '--preprocessed', config['preprocessed'], '--batch-size', str(args.batch_size)]
    if args.limit is not None:
        common += ['--limit', str(args.limit)]
    for step, script, kind in [('baseline', 'run_baseline.py', 'baseline'),
                                ('permutations', 'run_permutations.py', 'permutations'),
                                ('prior', 'estimate_prior.py', 'prior-estimation'),
                                ('pride', 'run_pride.py', 'pride')]:
        run = PROJECT_ROOT / 'outputs' / 'raw' / kind / directory.name
        run.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(PROJECT_ROOT / 'scripts' / script), *common, '--resume', str(run)]
        if step == 'prior':
            prior = directory / f'prior-{uuid.uuid4().hex[:12]}.json'
            command += ['--alpha', '0.05', '--output', str(prior)]
        if step == 'pride':
            command += ['--prior', state['prior'], '--baseline-run', state['steps']['baseline']]
        state['active_step'] = step
        write_json(state_path, state, overwrite=True)
        try:
            with (directory / f'{step}.log').open('a') as log:
                subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            state['status'] = 'failed'
            write_json(state_path, state, overwrite=True)
            print(f'{step}: FAILED. See {directory / (step + ".log")}', flush=True)
            raise
        state['steps'][step] = str(run)
        if step == 'prior':
            state['prior'] = str(prior)
        write_json(state_path, state, overwrite=True)
        print(f'{step}: OK ({run})', flush=True)
    state['active_step'] = 'aggregation'
    write_json(state_path, state, overwrite=True)
    subprocess.run([sys.executable, str(PROJECT_ROOT / 'scripts/aggregate_results.py'),
                    '--pipeline-id', directory.name, '--scorer', args.scorer,
                    '--output', str(directory / 'comparison.csv')], check=True, cwd=PROJECT_ROOT)
    state.update(status='completed', active_step=None)
    write_json(state_path, state, overwrite=True)

if __name__ == '__main__':
    main()

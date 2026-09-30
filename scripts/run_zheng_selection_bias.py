"""Run the pinned Zheng open-model conditions through their original functions.

No chat-template substitution. Real inference requires a Slurm GPU allocation.
Outputs are separate from the medical four-option pipeline.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import uuid

from _bootstrap import PROJECT_ROOT
from src.experimental.zheng_selection_bias import (
    COMMIT, HASHES, RecordingModel, load_original, option_diagnostic,
    original_settings, prepared, source_digest, summarize,
)
from src.utils.config import named_config
from src.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='qwen2.5-7b-instruct')
    p.add_argument('--dataset', choices=['mmlu', 'arc', 'csqa'], required=True)
    p.add_argument('--shots', nargs='+', type=int, choices=[0, 5], default=[0, 5])
    p.add_argument('--settings', nargs='+', choices=['default', 'perm', 'cyclic', 'shuffle_both',
                   'noid', 'move_a', 'move_b', 'move_c', 'move_d'])
    p.add_argument('--limit', type=int, help='Total questions across subjects; technical test only')
    p.add_argument('--resume', type=Path)
    p.add_argument('--plan-only', action='store_true', help='No model load or output writes')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    settings = args.settings or original_settings(args.dataset)
    if args.dataset == 'csqa' and 'perm' in settings:
        p.error('Original CSQA uses five cyclic permutations, not full permutations')
    model_config = named_config('models', args.model)
    config = dict(model=args.model, dataset=args.dataset, shots=list(dict.fromkeys(args.shots)),
                  settings=list(dict.fromkeys(settings)), limit=args.limit, upstream_commit=COMMIT,
                  source_hashes=HASHES, data_sha256=source_digest(args.dataset), model_config=model_config,
                  protocol='original-open-clm-raw-prompt-1536', scorer_backend='real_transformers')
    if args.plan_only:
        print(json.dumps(config, indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Real model execution requires a Slurm GPU job')
    if args.limit is None and any(model_config.get(key) in {None, 'main'}
                                 for key in ['model_revision', 'tokenizer_revision']):
        p.error('Pin model_revision and tokenizer_revision before full reference runs')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        p.error('CUDA GPU unavailable; no CPU fallback')
    directory = (args.resume or PROJECT_ROOT / 'outputs/zheng_selection_bias' /
                 f'{args.model}_{args.dataset}_{uuid.uuid4().hex[:12]}').resolve()
    if args.resume:
        state = json.loads((directory / 'manifest.json').read_text())
        if state['config'] != config:
            raise ValueError('Resume configuration differs')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        state = dict(config=config, status='initializing')
        write_json(directory / 'manifest.json', state)
    print(f'Run: {directory}', flush=True)
    original = load_original()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.get('tokenizer_repo', model_config['model_repo']),
            revision=model_config['tokenizer_revision'], use_fast=False,
            add_bos_token=False, add_eos_token=False, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_config['model_repo'], revision=model_config['model_revision'],
            device_map='auto', use_safetensors=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        model.eval()
        runtime = dict(model_revision=getattr(model.config, '_commit_hash', None),
                       tokenizer_revision=tokenizer.init_kwargs.get('_commit_hash'),
                       model_class=type(model).__name__, tokenizer_class=type(tokenizer).__name__,
                       dtype=str(model.dtype), gpu=torch.cuda.get_device_name(),
                       versions={key: importlib.metadata.version(key) for key in ['torch','transformers','numpy','pandas']})
        if args.resume and state.get('runtime') != runtime:
            raise ValueError('Runtime differs; do not mix model/package revisions on resume')
        labels = 'ABCDE' if args.dataset == 'csqa' else 'ABCD'
        diagnostic = option_diagnostic(tokenizer, labels)
        state.update(runtime=runtime, diagnostic=diagnostic, status='running', job_id=os.environ['SLURM_JOB_ID'])
        write_json(directory / 'manifest.json', state, overwrite=True)
        recorder = RecordingModel(model, diagnostic['slot_ids'])
        # Fixed question set shared by every condition. Filtering remains post-hoc.
        with prepared(original, args.dataset, 0, 'default') as (subjects, _, samples_fn, _):
            keys = [(subject, index) for subject in subjects for index in range(len(samples_fn(subject)))]
        chosen = {key for _, key in original.utils._index_samples(keys, args.limit)}
        raw_path = directory / 'records.jsonl'
        records = [json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
        completed = {(r['shots'], r['setting'], r['subject'], r['data']['idx']) for r in records}
        with raw_path.open('a') as stream:
            for shots in config['shots']:
                for setting in config['settings']:
                    state['active_condition'] = dict(shots=shots, setting=setting)
                    write_json(directory / 'manifest.json', state, overwrite=True)
                    with prepared(original, args.dataset, shots, setting) as (
                            subjects, fewshot_fn, samples_fn, evaluator_fn):
                        for subject in subjects:
                            wanted = {idx for subj, idx in chosen if subj == subject}
                            if not wanted:
                                continue
                            evaluator = evaluator_fn(recorder, tokenizer, fewshot_fn(subject))
                            for index, sample in enumerate(samples_fn(subject)):
                                key = (shots, setting, subject, index)
                                if index not in wanted or key in completed:
                                    continue
                                recorder.trace = []
                                record = evaluator((index, sample), random.Random(f'{index}:20230101'.encode()))
                                probabilities = np_array(record['data']['probs'])
                                if not probabilities:
                                    raise ValueError(f'Non-finite probabilities in {key}')
                                record.update(shots=shots, setting=setting, subject=subject,
                                              trace=recorder.trace)
                                stream.write(json.dumps(record, allow_nan=False) + '\n')
                                stream.flush()
                                records.append(record)
                                completed.add(key)
                    print(f'{shots}-shot {setting}: done', flush=True)
        expected = len(chosen) * len(config['shots']) * len(config['settings'])
        if len(completed) != expected:
            raise ValueError(f'Incomplete run: {len(completed)} / {expected}')
        summarize(records, args.dataset, original).to_csv(directory / 'summary.csv', index=False)
        state.update(status='completed', active_condition=None, selected_questions=len(chosen),
                     raw_record_count=len(records))
        write_json(directory / 'manifest.json', state, overwrite=True)
        print(f'Completed: {directory / "summary.csv"}', flush=True)
    except BaseException:
        state['status'] = 'failed'
        write_json(directory / 'manifest.json', state, overwrite=True)
        raise


def np_array(values):
    import numpy as np
    values = np.asarray(values)
    return bool(np.isfinite(values).all() and (values >= 0).all() and
                np.allclose(values.sum(axis=-1), 1, atol=1e-3))


if __name__ == '__main__':
    main()

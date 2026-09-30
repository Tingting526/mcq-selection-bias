"""Isolated access to Zheng's pinned open-model selection-bias implementation.

The original prompt/scoring functions are used unchanged. No GPU work on import.
Temporary working directories accommodate the upstream relative data paths.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'data/reference/zheng2024'
COMMIT = '2ae2f40c77006f7a00a4675bdfb30151c2406691'
HASHES = {
    'mmlu_categories': '070dea4c542cb029b6a6f6397da3c5c6fe95fd83a26d47e18cf79ffa18f13fb9',
    'utils': 'db451c3ca68216091a9ad6d2ecf0e754af5c43afe3b57dd8c906ad7e41daba6d',
    'eval_clm_utils': 'f48c8c36e60c15972c11b46712f0ce15c0ea8ca21f11eacb10602f9aa4bd7afa',
}


def load_original():
    """Verify source bytes and isolate the legacy top-level module names."""
    previous = {name: sys.modules.get(name) for name in HASHES}
    modules = {}
    try:
        for name, digest in HASHES.items():
            path = REFERENCE / 'upstream/code' / (name + '.py')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Original source changed: {path}')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return SimpleNamespace(evaluation=modules['eval_clm_utils'], utils=modules['utils'],
                           categories=modules['mmlu_categories'])


def original_settings(dataset):
    if dataset not in {'mmlu', 'arc', 'csqa'}:
        raise ValueError(dataset)
    # Directly mirrors scripts/run_llama-7b.sh; identity permutation is baseline.
    return (['cyclic'] if dataset == 'csqa' else ['perm']) + ['noid', 'shuffle_both'] + (
        ['move_a', 'move_b', 'move_c', 'move_d'] if dataset == 'mmlu' else [])


@contextmanager
def prepared(original, dataset, shots, setting):
    """Expose original callbacks without writing into the reference download.

    Changes cwd temporarily: use sequentially, not from multiple threads.
    """
    if dataset not in {'mmlu', 'arc', 'csqa'} or shots not in {0, 5}:
        raise ValueError('Expected original dataset and 0 or 5 shots')
    allowed = {'default', 'noid', 'shuffle_both', 'cyclic', 'perm',
               'move_a', 'move_b', 'move_c', 'move_d'}
    if setting not in allowed or (dataset == 'csqa' and setting == 'perm'):
        raise ValueError('Unsupported original condition')
    with tempfile.TemporaryDirectory(prefix='mcq-zheng-') as directory:
        old = Path.cwd()
        try:
            Path(directory, f'data_{dataset}').symlink_to(
                REFERENCE / f'upstream/code/data_{dataset}', target_is_directory=True)
            os.chdir(directory)
            name = f'{dataset},{shots}' + ('' if setting == 'default' else ',' + setting)
            yield original.evaluation.prepare_eval(SimpleNamespace(model_name='reference'), name)
        finally:
            os.chdir(old)


def excluded(options, original):
    return (any(x in option for x in original.utils.BAD_OPTIONS for option in options) or
            any(x in option.lower() for x in original.utils.REFER_OPTIONS for option in options))


def source_digest(dataset):
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    if manifest['commit'] != COMMIT:
        raise ValueError('Reference data commit differs')
    entries = [entry for entry in manifest['datasets'] if entry['dataset'] == dataset]
    if not entries:
        raise ValueError(f'Unknown reference dataset: {dataset}')
    for entry in entries:
        for item in entry['source_files']:
            path = REFERENCE / item['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Reference data changed: {path}')
    digest = hashlib.sha256()
    for path in sorted((REFERENCE / f'upstream/code/data_{dataset}').glob('*/*.csv')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def summarize(records, dataset, original):
    """Paper-style percent accuracy/recall/RStd, plus explicitly separate positions.

    Gold-moving conditions have one gold class, so their RStd is undefined.
    Filtering is applied at reporting time, as in upstream utils.load_results.
    """
    output = []
    for condition in sorted({(r['shots'], r['setting']) for r in records}):
        shots, setting = condition
        selected = [r for r in records if (r['shots'], r['setting']) == condition
                    and not excluded(r['data']['options'], original)]
        domains = ['Overall'] + (list(original.categories.categories) if dataset == 'mmlu' else [])
        for domain in domains:
            rows = [r for r in selected if domain == 'Overall' or
                    original.categories.subject2cat.get(r['subject']) == domain]
            if not rows:
                continue
            labels = 'ABCDE' if dataset == 'csqa' else 'ABCD'
            pred, gold, displayed = [], [], []
            for record in rows:
                data = record['data']
                if setting in {'perm', 'cyclic'}:
                    label = labels[int(np.argmax(data['probs'][0]))]
                else:
                    label = data['sampled']
                ideal = data['ideal']
                pred.append(label)
                gold.append(ideal)
                order = list(labels)
                if setting == 'shuffle_both':
                    order, _ = original.utils.shuffle_options_with_ids(order, data['options'])
                displayed.append(order.index(label))
            recalls = [100 * np.mean([p == label for p, g in zip(pred, gold) if g == label])
                       if label in gold else float('nan') for label in labels]
            result = dict(dataset=dataset, shots=shots, setting=setting, domain=domain,
                          n=len(rows), accuracy=100 * np.mean(np.array(pred) == np.array(gold)),
                          rstd=float(np.std(recalls)),
                          prediction_basis='option_content_likelihood' if setting == 'noid' else 'option_id_token',
                          baseline_from_identity=setting in {'perm', 'cyclic'})
            for i, label in enumerate(labels):
                result[f'recall_{label}'] = recalls[i]
                result[f'frequency_{label}'] = 100 * pred.count(label) / len(rows)
                result[f'displayed_position_{i+1}'] = 100 * displayed.count(i) / len(rows)
            output.append(result)
    return pd.DataFrame(output)


class RecordingModel:
    """Pass through to the real model; save only the option logits and token counts."""
    def __init__(self, model, option_ids):
        self.model = model
        self.device = model.device
        self.option_ids = option_ids
        self.trace = []

    def __call__(self, **kwargs):
        result = self.model(**kwargs)
        item = dict(input_tokens=int(kwargs['input_ids'].shape[-1]))
        if 'labels' in kwargs:
            item['loss'] = float(result.loss.detach().float().cpu())
            item['scored_tokens_after_truncation'] = int((kwargs['labels'][..., 1:] != -100).sum())
        else:
            item['option_slot_logits'] = result.logits[0, -1, self.option_ids].detach().float().cpu().tolist()
        self.trace.append(item)
        return result


def option_diagnostic(tokenizer, labels):
    """Keep the original 2*n slots, including duplicate token IDs, unchanged."""
    slots = [tokenizer(f': {label}').input_ids[-1] for label in labels] + [
        tokenizer(f':{label}').input_ids[-1] for label in labels]
    return dict(labels=list(labels), slot_ids=slots,
                slot_texts=[tokenizer.decode([i]) for i in slots],
                whitespace_variants_pooled=True, duplicate_slots_preserved=True,
                prompt_adds_space=tokenizer(': A').input_ids[-1] == tokenizer(':A').input_ids[-1])

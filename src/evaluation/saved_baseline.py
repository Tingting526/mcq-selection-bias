"""Reuse immutable first-token observations for paired before/after evaluation."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from src import OPTION_LABELS
from src.utils.io import RunStore, read_json


def records_digest(rows):
    frame = pd.DataFrame(rows).sort_values('sample_id').reset_index(drop=True)
    payload = frame.reindex(sorted(frame.columns), axis=1).to_json(orient='records', double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def reuse_saved_baseline(source, target, prior, samples, *, expected):
    source = Path(source).resolve()
    if source == target.run_dir.resolve():
        raise ValueError('Baseline source and PriDe destination must differ')
    metadata = read_json(source / 'metadata.json')
    mismatch = [key for key, value in expected.items() if metadata.get(key) != value]
    prior_metadata = read_json(Path(prior['raw_run_directory']) / 'metadata.json')
    for key in ('model_repo', 'scorer_backend', 'model_revision_resolved',
                'tokenizer_revision_resolved', 'option_token_ids',
                'prompt_template_version', 'chat_template_kwargs'):
        if metadata.get(key) != prior_metadata.get(key):
            mismatch.append('prior.' + key)
    if mismatch:
        raise ValueError('Saved baseline provenance mismatch: ' + ', '.join(mismatch))
    rows = RunStore(source).read_records()
    indexed = {str(row['sample_id']): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {str(s.sample_id) for s in samples}:
        raise ValueError('Saved baseline must contain each selected sample exactly once')
    for sample in samples:
        row = indexed[str(sample.sample_id)]
        probs = np.array([row[f'prob_{label}'] for label in OPTION_LABELS], dtype=float)
        if (not np.isfinite(probs).all() or (probs < 0).any()
                or not np.isclose(probs.sum(), 1.0)):
            raise ValueError('Invalid saved baseline probabilities')
        if (row['gold_label'] != sample.gold_label or int(row['permutation_id']) != 0
                or list(row['options']) != list(sample.options)
                or row['predicted_label'] != OPTION_LABELS[int(probs.argmax())]
                or bool(row['correct']) != (row['predicted_label'] == sample.gold_label)):
            raise ValueError('Saved baseline record does not match selected sample or scores')
    digest = records_digest(rows)
    metadata = {**metadata, 'baseline_source_run': str(source),
                'baseline_source_sha256': digest,
                'estimation_fraction_alpha': float(prior['alpha']),
                'prior_estimation_sample_ids': list(prior['estimation_sample_ids']),
                'prior_source_dataset': prior['source_dataset'],
                'prior_global': prior['global_prior']}
    # Old independent PriDe runs cannot be silently relabelled as paired runs.
    existing = target.read_records()
    if existing and records_digest(existing) != digest:
        raise ValueError('Existing PriDe inputs differ from original baseline; start a new pipeline')
    target.save_metadata(metadata)
    if not existing:
        target.write_records(rows)
    print(f'Reused {len(rows)} original baseline predictions; no second model pass')

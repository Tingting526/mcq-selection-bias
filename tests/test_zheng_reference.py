"""Numerical agreement with the inspected, pinned upstream PriDe functions."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from src.pride.prior_estimation import estimate_sample_prior, estimate_global_prior
from src.pride.debias import apply_pride_correction, cyclic_content_average

ROOT = Path(__file__).resolve().parents[1] / 'data/reference/zheng2024'


def test_pride_matches_upstream_for_positive_probabilities():
    path = ROOT / 'upstream/code/debias_utils.py'
    if not path.exists():
        pytest.skip('Download pinned reference with scripts/download_zheng2024.py')
    spec = importlib.util.spec_from_file_location('zheng_debias_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(42)
    mappings = [[(j+i) % 4 for j in range(4)] for i in range(4)]
    ours, theirs = [], []
    for _ in range(100):
        observed = rng.uniform(0.01, 1, size=(4, 4))
        observed /= observed.sum(axis=1, keepdims=True)
        _, debiased, prior = module.simple(observed)
        ours.append(estimate_sample_prior(observed))
        theirs.append(prior)
        np.testing.assert_allclose(ours[-1], prior, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(cyclic_content_average(observed, mappings), debiased,
                                   rtol=1e-7, atol=1e-9)
    global_prior = estimate_global_prior(ours)
    reference_prior = np.mean(theirs, axis=0)
    np.testing.assert_allclose(global_prior, reference_prior, rtol=1e-7, atol=1e-9)
    for _ in range(100):
        observed = rng.dirichlet(np.ones(4))
        corrected = apply_pride_correction(observed, global_prior)
        ref = module.softmax(np.log(observed + 1e-10) - np.log(reference_prior + 1e-10))
        np.testing.assert_allclose(corrected, ref, rtol=1e-7, atol=1e-9)
        assert corrected.argmax() == ref.argmax()


def test_reference_counts_and_five_options_preserved():
    if not (ROOT / 'manifest.json').exists():
        pytest.skip('Reference datasets not downloaded')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    counts = {row['dataset']: row['retained_rows'] for row in manifest['datasets']
              if row['repository_split'] == 'test'}
    assert counts == {'mmlu': 13592, 'arc': 1165, 'csqa': 1216}
    records = [json.loads(line) for line in (ROOT / 'normalized/csqa_test.jsonl').read_text().splitlines()]
    assert all(len(row['options']) == 5 for row in records)
    assert {row['gold_index'] for row in records} == set(range(5))

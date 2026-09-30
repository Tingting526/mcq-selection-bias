from pathlib import Path
import sys
import pandas as pd
import pytest
from src.utils.io import RunStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from aggregate_results import aggregate


def test_resume_rejects_mock_real_mix(tmp_path):
    store = RunStore(tmp_path / 'run')
    store.save_metadata({'scorer_backend': 'mock'})
    with pytest.raises(ValueError, match='scorer_backend'):
        store.save_metadata({'scorer_backend': 'real_transformers'})


def test_aggregation_requires_complete_matching_pipeline(tmp_path):
    common = dict(pipeline_id='full', model_repo='m', dataset_repo='d',
                  scorer_backend='real_transformers', seed=42, sample_set_sha256='abc')
    for kind in ['baseline', 'permutations']:
        pd.DataFrame([{**common, 'kind': kind, 'run_dir': kind, 'accuracy': 0.5}]).to_csv(tmp_path / f'{kind}.csv', index=False)
    with pytest.raises(ValueError, match='Incomplete'):
        aggregate(tmp_path, 'full', 'real')
    pd.DataFrame([{**common, 'kind': 'pride', 'run_dir': 'pride', 'delta_accuracy': 0.1}]).to_csv(tmp_path / 'pride.csv', index=False)
    result = aggregate(tmp_path, 'full', 'real')
    assert len(result) == 1
    assert result.iloc[0]['pride_delta_accuracy'] == 0.1
    with pytest.raises(ValueError, match='No matching'):
        aggregate(tmp_path, 'full', 'mock')


def test_full_campaign_clears_inherited_smoke_settings(tmp_path, monkeypatch):
    from run_campaign import plan, job_environment
    import run_campaign
    monkeypatch.setattr(run_campaign, 'artifact_for', lambda dataset: tmp_path / dataset)
    monkeypatch.setenv('LIMIT', '8')
    monkeypatch.setenv('STEP', 'baseline')
    monkeypatch.setenv('PRIOR', 'old-prior.json')
    first = plan(tmp_path / 'first', ['qwen3-8b'], ['medqa'])
    second = plan(tmp_path / 'second', ['qwen3-8b'], ['medqa'])
    entry = first['entries'][0]
    env = job_environment(entry)
    assert 'LIMIT' not in env and 'PRIOR' not in env
    assert env['STEP'] == 'pipeline'
    assert env['RESUME'] == entry['pipeline']
    assert Path(entry['pipeline']).name != Path(second['entries'][0]['pipeline']).name


def test_campaign_submission_does_not_duplicate_jobs(tmp_path, monkeypatch):
    import run_campaign
    from types import SimpleNamespace
    monkeypatch.setattr(run_campaign, 'artifact_for', lambda dataset: tmp_path / dataset)
    state = run_campaign.plan(tmp_path / 'campaign', ['qwen3-8b'], ['medqa'], limit=8)
    calls = []
    def submit_stub(*args, **kwargs):
        calls.append(kwargs['env'])
        return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    monkeypatch.setattr(run_campaign.subprocess, 'run', submit_stub)
    run_campaign.submit(tmp_path / 'campaign', state, 'example-gpu')
    run_campaign.submit(tmp_path / 'campaign', state, 'example-gpu')
    assert len(calls) == 1
    assert calls[0]['LIMIT'] == '8'
    assert state['entries'][0]['job_id'] == '12345'


def test_campaign_collection_rejects_unfinished_runs(tmp_path, monkeypatch):
    import run_campaign
    monkeypatch.setattr(run_campaign, 'artifact_for', lambda dataset: tmp_path / dataset)
    state = run_campaign.plan(tmp_path / 'campaign', ['qwen3-8b'], ['medqa'])
    with pytest.raises(ValueError, match='not completed'):
        run_campaign.collect(tmp_path / 'campaign', state)

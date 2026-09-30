import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.metrics.selection_bias import paired_group_metrics
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_campaign
import start_medqa_campaign


def test_paired_groups_use_ids_not_order():
    baseline = [dict(sample_id='a',gold_label='A',predicted_label='B'),
                dict(sample_id='b',gold_label='B',predicted_label='B')]
    after = [dict(sample_id='b',gold_label='B',predicted_label='A',sample_group='D_r'),
             dict(sample_id='a',gold_label='A',predicted_label='A',sample_group='D_e')]
    result = paired_group_metrics(baseline, after)
    assert result['D_e_baseline_n'] == 1
    assert result['D_e_delta_accuracy'] == 1
    assert result['D_r_delta_accuracy'] == -1
    with pytest.raises(ValueError, match='unique sample IDs'):
        paired_group_metrics(baseline, after[:1])


def test_full_medqa_uses_explicit_artifact_for_all_five(tmp_path, monkeypatch):
    monkeypatch.setattr(run_campaign, 'artifact_for', lambda _: pytest.fail('Ambiguous discovery used'))
    state = run_campaign.plan(tmp_path/'campaign', start_medqa_campaign.MODELS, ['medqa'],
                              preprocessed=tmp_path/'artifact')
    assert len(state['entries']) == 5
    for entry in state['entries']:
        assert entry['config']['limit'] is None
        assert entry['config']['preprocessed'] == str(tmp_path/'artifact')


def test_failed_model_access_does_not_submit(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['start_medqa_campaign.py','--submit'])
    monkeypatch.setattr(start_medqa_campaign, 'select_artifact', lambda: tmp_path)
    monkeypatch.setattr(start_medqa_campaign, 'check_access', lambda _: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(start_medqa_campaign, 'submit', lambda *a: pytest.fail('Submitted with failed access'))
    with pytest.raises(SystemExit, match='Nothing submitted'):
        start_medqa_campaign.main()


def test_selected_models_only_checked_and_submitted(tmp_path, monkeypatch):
    selected = ['qwen3-8b', 'mistral-7b-instruct-v0.3', 'glm-4-9b-chat-hf']
    monkeypatch.setattr(sys, 'argv', ['start_medqa_campaign.py', '--submit', '--directory',
                                   str(tmp_path/'campaign'), '--models', *selected])
    monkeypatch.setattr(start_medqa_campaign, 'select_artifact', lambda: tmp_path/'artifact')
    checked, submitted = [], []
    monkeypatch.setattr(start_medqa_campaign, 'check_access', checked.append)
    monkeypatch.setattr(start_medqa_campaign, 'submit', lambda directory,state,partition: submitted.extend(state['entries']))
    start_medqa_campaign.main()
    assert checked == selected
    assert [e['config']['model'] for e in submitted] == selected
    assert all(e['config']['limit'] is None for e in submitted)

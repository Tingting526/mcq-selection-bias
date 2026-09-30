from types import SimpleNamespace
import numpy as np
import torch
import pytest
from src.experimental.zheng_selection_bias import (
    load_original, prepared, summarize, option_diagnostic, original_settings, RecordingModel,
)

class ToyTokenizer:
    """Distinct space/no-space option IDs, no model/download."""
    padding_side = 'left'
    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(c) % 120 + 1 for c in text]
        if len(text) >= 2 and text[-1] in 'ABCDE' and text[-2] == ' ':
            ids[-1] += 120
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)
    def decode(self, ids):
        return str(ids)

class ToyModel:
    device = torch.device('cpu')
    def __call__(self, input_ids, labels=None):
        logits = torch.linspace(-2, 2, 256).repeat(1, input_ids.shape[-1], 1)
        loss = torch.tensor(0.5) if labels is None else labels[labels != -100].float().mean() / 100
        return SimpleNamespace(logits=logits, loss=loss)

@pytest.fixture(scope='module')
def original():
    return load_original()

def test_original_conditions_are_not_only_cyclic():
    assert original_settings('mmlu') == ['perm', 'noid', 'shuffle_both', 'move_a', 'move_b', 'move_c', 'move_d']
    assert original_settings('csqa') == ['cyclic', 'noid', 'shuffle_both']

def test_first_token_pools_original_two_variants(original):
    tokenizer = ToyTokenizer()
    diagnostic = option_diagnostic(tokenizer, 'ABCDE')
    model = RecordingModel(ToyModel(), diagnostic['slot_ids'])
    fn = original.evaluation.prepare_eval_fn_base(model, tokenizer, [], 0, list('ABCDE'))
    row = fn((0, (['System', 'Question?\nAnswer:'], ['a','b','c','d','e'], 'A')), None)
    logits = torch.linspace(-2, 2, 256)[diagnostic['slot_ids']]
    expected = torch.softmax(logits, dim=-1).reshape(2, 5).sum(0).numpy()
    np.testing.assert_allclose(row['data']['probs'], expected)
    assert len(model.trace[0]['option_slot_logits']) == 10
    assert len(set(diagnostic['slot_ids'])) == 10

def test_original_moves_swap_gold_and_shuffling_preserves_labels(original):
    with prepared(original, 'arc', 0, 'default') as (_, _, samples_fn, _):
        baseline = samples_fn('arc')[0]
    target = 'A' if baseline[2] != 'A' else 'B'
    with prepared(original, 'arc', 0, 'move_' + target.lower()) as (_, _, samples_fn, _):
        moved = samples_fn('arc')[0]
    old_gold, new_gold = 'ABCD'.index(baseline[2]), 'ABCD'.index(target)
    assert moved[1][new_gold] == baseline[1][old_gold]
    assert moved[1][old_gold] == baseline[1][new_gold]
    assert moved[2] == target
    ids, options = original.utils.shuffle_options_with_ids(list('ABCD'), baseline[1])
    assert dict(zip(ids, options)) == dict(zip('ABCD', baseline[1]))

@pytest.mark.parametrize('dataset,k', [('arc',4), ('csqa',5)])
def test_every_original_condition_runs_model_free(original, dataset, k):
    records = []
    for setting in original_settings(dataset):
        for shots in [0, 5]:
            with prepared(original, dataset, shots, setting) as (_, fewshot_fn, samples_fn, make_fn):
                tokenizer = ToyTokenizer()
                model = RecordingModel(ToyModel(), option_diagnostic(tokenizer, 'ABCDE'[:k])['slot_ids'])
                fn = make_fn(model, tokenizer, fewshot_fn(dataset))
                record = fn((0, samples_fn(dataset)[0]), None)
                record.update(setting=setting, shots=shots, subject=dataset)
                records.append(record)
                if setting == 'noid':
                    assert len(record['data']['losses']) == k
                    assert len(model.trace) == k
                    assert all('loss' in item for item in model.trace)
                elif setting in {'perm','cyclic'}:
                    assert np.asarray(record['data']['probs']).shape == ((24,4) if k == 4 else (5,5))
    frame = summarize(records, dataset, original)
    assert len(frame) == 6
    assert frame.n.eq(1).all()
    assert frame.rstd.isna().all()

def test_summary_filter_recall_and_position_are_separate(original):
    records = [dict(shots=0, setting='shuffle_both', subject='arc', data=dict(
        options=['one','two','three','four'], ideal=label, sampled='A', idx=i))
        for i,label in enumerate('ABCD')]
    records.append(dict(shots=0, setting='shuffle_both', subject='arc', data=dict(
        options=['one','two','three','none of the above'], ideal='A', sampled='A', idx=4)))
    row = summarize(records,'arc',original).iloc[0]
    assert row.n == 4 and row.accuracy == 25
    assert row.frequency_A == 100
    assert row.rstd == pytest.approx(np.std([100,0,0,0]))
    assert sum(row[f'displayed_position_{i}'] for i in range(1,5)) == 100

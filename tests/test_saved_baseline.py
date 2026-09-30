from types import SimpleNamespace
import pytest
from src.evaluation.saved_baseline import reuse_saved_baseline
from src.utils.io import RunStore


def fixture(tmp_path):
    source, prior_store, target = [RunStore(tmp_path / name) for name in ('baseline', 'prior', 'pride')]
    metadata = {'model_repo': 'model', 'option_token_ids': [1, 2, 3, 4]}
    source.save_metadata(metadata)
    prior_store.save_metadata(metadata)
    row = dict(sample_id='q', gold_label='B', predicted_label='A', correct=False,
               permutation_id=0, options=['one', 'two', 'three', 'four'],
               prob_A=.4, prob_B=.3, prob_C=.2, prob_D=.1)
    source.write_records([row])
    prior = dict(raw_run_directory=str(prior_store.run_dir), alpha=.05,
                 estimation_sample_ids=['q'], source_dataset='dataset', global_prior=[.25]*4)
    samples = [SimpleNamespace(sample_id='q', gold_label='B', options=row['options'])]
    return source, target, prior, samples


def test_reuse_preserves_predictions_and_resume_is_idempotent(tmp_path):
    source, target, prior, samples = fixture(tmp_path)
    for _ in range(2):
        reuse_saved_baseline(source.run_dir, target, prior, samples, expected={'model_repo': 'model'})
    assert len(target.read_records()) == 1
    assert target.read_records()[0]['prob_A'] == .4
    assert target.read_records()[0]['predicted_label'] == 'A'
    assert len(source.part_files()) == 1


def test_reuse_rejects_different_token_mapping(tmp_path):
    source, target, prior, samples = fixture(tmp_path)
    from src.utils.io import write_json
    write_json(RunStore(prior['raw_run_directory']).run_dir / 'metadata.json',
               {'model_repo': 'model', 'option_token_ids': [5,6,7,8]}, overwrite=True)
    with pytest.raises(ValueError, match='option_token_ids'):
        reuse_saved_baseline(source.run_dir, target, prior, samples, expected={})
    assert not target.part_files()


def test_reuse_rejects_old_different_pride_baseline(tmp_path):
    source, target, prior, samples = fixture(tmp_path)
    row = source.read_records()[0]
    row['prob_A'] = .5
    target.write_records([row])
    with pytest.raises(ValueError, match='start a new pipeline'):
        reuse_saved_baseline(source.run_dir, target, prior, samples, expected={})


def test_reuse_rejects_incomplete_sample_population(tmp_path):
    source, target, prior, samples = fixture(tmp_path)
    samples.append(SimpleNamespace(sample_id='missing'))
    with pytest.raises(ValueError, match='exactly once'):
        reuse_saved_baseline(source.run_dir, target, prior, samples, expected={})

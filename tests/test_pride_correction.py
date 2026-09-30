import numpy as np

from src.data.base import MCQSample
from src.permutations.cyclic import cyclic_permutations
from src.pride.debias import apply_pride_correction, cyclic_content_average


def test_pride_division_recovers_content_distribution() -> None:
    content = np.asarray([0.55, 0.25, 0.15, 0.05])
    prior = np.asarray([0.4, 0.3, 0.2, 0.1])
    observed = prior * content
    observed /= observed.sum()
    corrected = apply_pride_correction(observed, prior)
    assert np.allclose(corrected, content)
    assert np.isclose(corrected.sum(), 1.0)


def test_cyclic_average_tracks_content_not_display_label() -> None:
    sample = MCQSample("q", "test", "Question", ("o0", "o1", "o2", "o3"), 0)
    permutations = cyclic_permutations(sample)
    content_probabilities = np.asarray([0.6, 0.2, 0.15, 0.05])
    displayed_rows = []
    mappings = []
    for permutation in permutations:
        displayed_rows.append(
            [content_probabilities[content] for content in permutation.display_to_content]
        )
        mappings.append(permutation.display_to_content)
    averaged = cyclic_content_average(displayed_rows, mappings)
    assert np.allclose(averaged, content_probabilities)


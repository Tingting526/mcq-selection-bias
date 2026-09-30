from src.data.base import MCQSample
from src.permutations.cyclic import cyclic_permutations


def test_cyclic_content_orders_are_exact() -> None:
    sample = MCQSample("q", "test", "Question", ("o0", "o1", "o2", "o3"), 0)
    permutations = cyclic_permutations(sample)
    assert [permutation.display_to_content for permutation in permutations] == [
        (0, 1, 2, 3),
        (1, 2, 3, 0),
        (2, 3, 0, 1),
        (3, 0, 1, 2),
    ]
    assert [permutation.displayed_options for permutation in permutations] == [
        ("o0", "o1", "o2", "o3"),
        ("o1", "o2", "o3", "o0"),
        ("o2", "o3", "o0", "o1"),
        ("o3", "o0", "o1", "o2"),
    ]


def test_each_content_appears_once_under_each_label() -> None:
    sample = MCQSample("q", "test", "Question", ("o0", "o1", "o2", "o3"), 0)
    permutations = cyclic_permutations(sample)
    for content_index in range(4):
        assert sorted(p.content_to_display[content_index] for p in permutations) == [0, 1, 2, 3]


def test_duplicate_option_text_does_not_break_identity_mapping() -> None:
    sample = MCQSample("q", "test", "Question", ("same", "same", "other", "last"), 1)
    permutations = cyclic_permutations(sample)
    assert [permutation.gold_display_index for permutation in permutations] == [1, 0, 3, 2]


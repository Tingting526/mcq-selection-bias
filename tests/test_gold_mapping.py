from src.data.base import MCQSample
from src.permutations.cyclic import cyclic_permutations


def test_gold_label_is_derived_from_gold_content_after_each_permutation() -> None:
    sample = MCQSample("q", "test", "Question", ("o0", "o1", "gold", "o3"), 2)
    permutations = cyclic_permutations(sample)
    assert [permutation.gold_display_label for permutation in permutations] == ["C", "B", "A", "D"]
    assert all(
        permutation.display_to_content[permutation.gold_display_index] == sample.gold_index
        for permutation in permutations
    )


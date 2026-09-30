"""Algorithm 1: estimation set D_e (cyclic debiasing) vs remaining set D_r (Eq. 8)."""
from src.evaluation.pride_eval import split_estimation_and_remaining


def _pride_row(sample_id, predicted="A"):
    row = {
        "sample_id": sample_id,
        "predicted_label": predicted,
        "gold_label": "B",
        "correct": predicted == "B",
        "original_prediction": predicted,
        "pride_prediction": predicted,
    }
    for label in "ABCD":
        row[f"corrected_prob_{label}"] = 0.25
    return row


def _cyclic_row(sample_id, predicted="C"):
    row = {"sample_id": sample_id, "predicted_label": predicted, "correct": predicted == "B"}
    for label in "ABCD":
        row[f"cyclic_prob_content_{label}"] = 0.7 if label == predicted else 0.1
    return row


def test_estimation_samples_get_cyclic_remaining_keep_eq8():
    pride = [_pride_row("s1", "A"), _pride_row("s2", "A")]
    cyclic = [_cyclic_row("s1", "C")]
    out = {r["sample_id"]: r for r in split_estimation_and_remaining(pride, cyclic, ["s1"])}

    assert out["s1"]["sample_group"] == "D_e"
    assert out["s1"]["debiasing_method"] == "cyclic_estimation_sample"
    assert out["s1"]["predicted_label"] == "C"
    assert out["s1"]["corrected_prob_C"] == 0.7

    assert out["s2"]["sample_group"] == "D_r"
    assert out["s2"]["debiasing_method"] == "pride_global_prior"
    assert out["s2"]["predicted_label"] == "A"  # unchanged Equation-8 prediction


def test_transferred_marks_every_sample_as_dr():
    out = split_estimation_and_remaining([_pride_row("t1")], [], ["s1"], transferred=True)
    assert out[0]["sample_group"] == "D_r"
    assert out[0]["debiasing_method"] == "pride_transferred_global_prior"

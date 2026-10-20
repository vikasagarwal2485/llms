import pytest

from classifier_model.metrics import classification_metrics


def test_three_classes_with_never_predicted_class():
    # true:  a a b b c c ; pred: a b b b a a  (class c is never predicted)
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 0, 0]
    m = classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(3 / 6)
    a, b, c = m["per_class"]
    assert a["precision"] == pytest.approx(1 / 3) and a["recall"] == pytest.approx(1 / 2)
    assert a["f1"] == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))
    assert b["precision"] == pytest.approx(2 / 3) and b["recall"] == pytest.approx(1.0)
    assert c == {"label": "c", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}
    assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 2, 0], [2, 0, 0]]

"""Classification metrics with numpy (no sklearn)."""

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    """Rows are true classes, columns are predicted classes."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(y_true, y_pred, label_names):
    matrix = confusion_matrix(y_true, y_pred, len(label_names))
    per_class = []
    for i, name in enumerate(label_names):
        true_positive = matrix[i, i]
        precision = _safe_divide(true_positive, matrix[:, i].sum())
        recall = _safe_divide(true_positive, matrix[i, :].sum())
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class.append({
            "label": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[i, :].sum()),
        })
    return {
        "accuracy": _safe_divide(np.trace(matrix), matrix.sum()),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(label_names), "matrix": matrix.tolist()},
    }

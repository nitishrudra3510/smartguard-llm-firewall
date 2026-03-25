# metrics.py — Evaluation metric helpers

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)


def compute_binary_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : list of int   1 = Unsafe, 0 = Safe (ground truth)
    y_pred : list of int   1 = Unsafe, 0 = Safe (predictions)

    Returns
    -------
    dict with accuracy, precision, recall, f1
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def compute_threshold_sweep(results: list, thresholds=None) -> list:
    """
    For each threshold value, compute accuracy when applying that threshold
    to confidence scores.

    Parameters
    ----------
    results   : list of dicts with keys label, confidence, true_label
    thresholds: iterable of float values to sweep

    Returns
    -------
    list of (threshold, accuracy) tuples
    """
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]   # 0.05 → 0.95

    sweep = []
    for t in thresholds:
        preds = []
        for r in results:
            if r["label"] == "Unsafe" and r["confidence"] > t:
                preds.append(1)
            else:
                preds.append(0)
        y_true = [1 if r["true_label"] == "Unsafe" else 0 for r in results]
        acc = accuracy_score(y_true, preds)
        sweep.append((round(t, 2), round(acc, 4)))

    return sweep


def full_report(y_true: list, y_pred: list, target_names=None) -> str:
    """Return sklearn's classification_report as a string."""
    return classification_report(
        y_true, y_pred,
        target_names=target_names or ["Safe", "Unsafe"],
        zero_division=0,
    )

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


def multiclass_mcc(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64, copy=False)
    total = cm.sum()
    if total <= 0:
        return float("nan")
    true_sum = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
    trace = np.trace(cm)
    numerator = trace * total - np.dot(true_sum, pred_sum)
    denominator = math.sqrt(
        max(total * total - np.dot(pred_sum, pred_sum), 0.0)
        * max(total * total - np.dot(true_sum, true_sum), 0.0)
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def confusion_metrics(cm: np.ndarray, class_names: list[str], benign_index: int | None) -> tuple[dict, list[dict]]:
    cm = cm.astype(np.float64, copy=False)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    recall = safe_divide(tp, support)
    precision = safe_divide(tp, predicted)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    active = support > 0
    total = support.sum()

    if active.any():
        active_recall = recall[active]
        active_f1 = f1[active]
        balanced_accuracy = float(active_recall.mean())
        macro_f1 = float(active_f1.mean())
        worst_class_recall = float(active_recall.min())
        gmean = 0.0 if np.any(active_recall <= 0) else float(np.exp(np.log(active_recall).mean()))
    else:
        balanced_accuracy = macro_f1 = worst_class_recall = gmean = float("nan")

    priors = safe_divide(support, np.full_like(support, total)) if total > 0 else np.zeros_like(support)
    minority = active & (priors < 1.0 / len(class_names))
    dynamic_minority_recall = float(recall[minority].mean()) if minority.any() else float("nan")
    attack_mask = active.copy()
    if benign_index is not None:
        attack_mask[benign_index] = False
    minority_attack = minority & attack_mask
    dynamic_minority_attack_recall = (
        float(recall[minority_attack].mean()) if minority_attack.any() else float("nan")
    )
    worst_attack_recall = float(recall[attack_mask].min()) if attack_mask.any() else float("nan")

    benign_fpr = float("nan")
    if benign_index is not None and support[benign_index] > 0:
        benign_fpr = float((support[benign_index] - cm[benign_index, benign_index]) / support[benign_index])

    summary = {
        "n": int(total),
        "accuracy": float(tp.sum() / total) if total > 0 else float("nan"),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "gmean": gmean,
        "mcc": multiclass_mcc(cm),
        "dynamic_minority_recall": dynamic_minority_recall,
        "dynamic_minority_attack_recall": dynamic_minority_attack_recall,
        "worst_class_recall": worst_class_recall,
        "worst_attack_recall": worst_attack_recall,
        "benign_fpr": benign_fpr,
        "active_classes": int(active.sum()),
    }

    class_rows: list[dict] = []
    for i, label in enumerate(class_names):
        class_rows.append(
            {
                "class": label,
                "support": int(support[i]),
                "predicted": int(predicted[i]),
                "precision": float(precision[i]) if predicted[i] > 0 else float("nan"),
                "recall": float(recall[i]) if support[i] > 0 else float("nan"),
                "f1": float(f1[i]) if support[i] > 0 else float("nan"),
                "prior": float(priors[i]) if total > 0 else float("nan"),
                "is_dynamic_minority": bool(minority[i]),
            }
        )
    return summary, class_rows


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@dataclass
class WindowAccumulator:
    max_blocks: int
    n_classes: int

    def __post_init__(self) -> None:
        self.blocks: deque[np.ndarray] = deque()
        self.total = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def append(self, cm: np.ndarray) -> None:
        copied = cm.astype(np.int64, copy=True)
        self.blocks.append(copied)
        self.total += copied
        if len(self.blocks) > self.max_blocks:
            self.total -= self.blocks.popleft()

    @property
    def ready(self) -> bool:
        return len(self.blocks) == self.max_blocks


class ModelMetricState:
    def __init__(self, n_classes: int, block_size: int, rolling_windows: Iterable[int]) -> None:
        self.n_classes = int(n_classes)
        self.block_size = int(block_size)
        self.block = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.cumulative = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.windows = {
            int(window): WindowAccumulator(int(window) // self.block_size, n_classes)
            for window in rolling_windows
        }

    def update(self, y_true: int, y_pred: int) -> None:
        self.block[y_true, y_pred] += 1

    def close_block(self) -> np.ndarray:
        completed = self.block
        self.cumulative += completed
        for window in self.windows.values():
            window.append(completed)
        self.block = np.zeros_like(completed)
        return completed

"""
evaluation.py
-------------
Reusable evaluation and plotting utilities extracted from the ML course exercises.

Covers:
- Classification metrics summary  (Exercises 1, 2, 4, 5, 7)
- Confusion matrix plot            (Exercises 2, 5, 7)
- Feature importance bar chart     (Exercises 2, 4)
- Keras learning-curve plot        (Exercise 8)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def print_metrics(y_true, y_pred, target_names=None, average: str = "weighted"):
    """Print a concise classification report.

    Parameters
    ----------
    y_true       : ground-truth labels
    y_pred       : predicted labels
    target_names : list of class name strings (optional)
    average      : averaging strategy for precision/recall/F1
                   ("weighted", "macro", "binary")
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}  ({average})")
    print(f"Recall    : {rec:.4f}  ({average})")
    print(f"F1-score  : {f1:.4f}  ({average})")
    print()
    print(classification_report(y_true, y_pred, target_names=target_names,
                                zero_division=0))


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, labels=None, title: str = "Confusion Matrix",
                          figsize: tuple = (7, 6), cmap: str = "Blues"):
    """Plot a labeled confusion matrix.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : predicted labels
    labels  : display names for the classes (optional)
    title   : plot title
    figsize : figure size
    cmap    : matplotlib colormap
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap=cmap)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(importances, feature_names, top_n: int = 15,
                             title: str = "Feature Importances", figsize: tuple = (9, 5)):
    """Horizontal bar chart of feature importances (e.g. from Random Forest).

    Parameters
    ----------
    importances   : array-like of importance scores
    feature_names : array-like of feature name strings
    top_n         : how many top features to display
    title         : plot title
    figsize       : figure size
    """
    importances = np.array(importances)
    feature_names = np.array(feature_names)

    n = min(top_n, len(importances))
    indices = np.argsort(importances)[::-1][:n]
    top_imp = importances[indices]
    top_names = feature_names[indices]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(n), top_imp[::-1], align="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Keras learning curves (Exercise 8)
# ---------------------------------------------------------------------------

def plot_learning_curves(history, metrics=("loss", "accuracy"), figsize: tuple = (12, 4)):
    """Plot training and validation curves from a Keras History object.

    Parameters
    ----------
    history : keras History object returned by model.fit()
    metrics : tuple of metric names to plot
    figsize : figure size
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history.history:
            ax.plot(history.history[metric], label=f"Train {metric}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f"Val {metric}", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()

    plt.tight_layout()
    plt.show()

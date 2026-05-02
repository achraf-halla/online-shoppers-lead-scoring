"""Evaluation helpers focused on what matters for lead scoring.

ROC-AUC alone is misleading on imbalanced data. We track:

* `pr_auc` — average precision; sensitive to the minority class.
* `roc_auc` — for completeness and comparability.
* `brier` — proper scoring rule; rewards calibrated probabilities.
* `top_k_precision` — what fraction of the top-scored leads are real.
  In a real campaign, sales has bandwidth for X% of the list, not 100%.
* `lift_curve_data` / `plot_cumulative_gains` — the chart that
  marketing actually understands.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def basic_metrics(y_true, y_score) -> dict[str, float]:
    """ROC-AUC, PR-AUC, Brier in one dict."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
    }


def top_k_precision(y_true, y_score, k_frac: float = 0.1) -> float:
    """Precision among the top `k_frac` fraction of scored examples.

    Equivalent to "if sales calls the top X% of leads, what fraction are
    actual converters?".
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = max(1, int(len(y_score) * k_frac))
    top_idx = np.argsort(y_score)[-n:]
    return float(y_true[top_idx].mean())


def lift_curve_data(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Per-decile cumulative gains, lift, and capture rate."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    df = (pd.DataFrame({"y": y_true, "score": y_score})
            .sort_values("score", ascending=False)
            .reset_index(drop=True))
    df["bin"] = pd.qcut(df.index, n_bins, labels=range(1, n_bins + 1))

    out = df.groupby("bin", observed=True).agg(
        n=("y", "size"),
        positives=("y", "sum"),
    )
    out["cum_n"] = out["n"].cumsum()
    out["cum_positives"] = out["positives"].cumsum()
    base_rate = df["y"].mean()
    out["cum_share_pop"] = out["cum_n"] / len(df)
    out["cum_share_pos"] = out["cum_positives"] / df["y"].sum()
    out["cum_precision"] = out["cum_positives"] / out["cum_n"]
    out["lift"] = out["cum_precision"] / base_rate
    return out


def plot_calibration(y_true,
                     scores: Mapping[str, np.ndarray],
                     n_bins: int = 10,
                     ax=None):
    """Reliability diagram for one or more model scores.

    `scores` maps a label (e.g. "LightGBM raw", "LightGBM isotonic") to
    a 1-D array of predicted probabilities of the same length as y_true.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    for label, s in scores.items():
        prob_true, prob_pred = calibration_curve(
            y_true, s, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical fraction positive")
    ax.set_title("Reliability diagram")
    ax.legend()
    return ax


def plot_cumulative_gains(y_true,
                          scores: Mapping[str, np.ndarray],
                          ax=None):
    """Cumulative gains: x = share of population scored, y = share of positives captured."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for label, s in scores.items():
        data = lift_curve_data(y_true, s)
        x = [0] + data["cum_share_pop"].tolist()
        y_vals = [0] + data["cum_share_pos"].tolist()
        ax.plot(x, y_vals, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    ax.set_xlabel("Share of population (sorted by score)")
    ax.set_ylabel("Share of positives captured")
    ax.set_title("Cumulative gains")
    ax.legend()
    return ax


def metrics_table(y_true,
                  scores: Mapping[str, np.ndarray],
                  k_fracs: tuple[float, ...] = (0.1, 0.2, 0.3)) -> pd.DataFrame:
    """One-row-per-model summary with PR-AUC, ROC-AUC, Brier, top-K precision."""
    rows = []
    for label, s in scores.items():
        row = {"model": label, **basic_metrics(y_true, s)}
        for k in k_fracs:
            row[f"prec@top{int(k*100)}%"] = top_k_precision(y_true, s, k_frac=k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(4)

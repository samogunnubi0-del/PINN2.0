"""Jackknife+ / CV+ conformal prediction (Barber, Candes, Ramdas, Tibshirani 2021).

Reference: R. F. Barber, E. J. Candes, A. Ramdas, R. J. Tibshirani,
"Predictive inference with the jackknife+", Annals of Statistics 49(1), 2021.
arXiv:1905.02928. CV+ is the K-fold analogue in the same paper.

Two levels are implemented, and the module is explicit about which guarantee
applies — this distinction matters for the ISEF write-up:

1. ``cv_plus_intervals`` — the EXACT CV+ / jackknife+ formula. It requires
   per-fold models (each trained without fold k) and their predictions on the
   calibration and test points. Given those, CV+ intervals carry the
   distribution-free 1 - 2*alpha marginal coverage guarantee of the paper.

2. ``frozen_cv_plus`` — the CHEAP variant used with a frozen checkpoint. All
   "fold models" are the same network, so the CV+ construction DEGENERATES to
   ordinary split conformal (verified numerically and reported as
   ``degenerates_to_split_conformal: true``). The strict 1 - 2*alpha
   jackknife+ guarantee does NOT apply here; what is inherited is the standard
   split-conformal marginal coverage (>= 1 - alpha up to finite-sample
   correction), and the fold/bootstrap machinery quantifies the STABILITY of
   the interval width (how much q moves when the calibration set changes) —
   itself a valuable judge-facing robustness number. For the true guarantee,
   retrain per fold and call ``cv_plus_intervals`` (a Kaggle-scale job).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1-alpha) quantile (same rule as split conformal)."""
    scores = np.asarray(scores, dtype=np.float64)
    n = int(scores.size)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def cv_plus_intervals(
    cal_y: np.ndarray,
    cal_preds_loo: np.ndarray,
    test_preds_loo: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """EXACT CV+ / jackknife+ intervals (Barber et al. 2021, Thm. 1 & 4).

    Args:
        cal_y:          (n_cal,) true calibration targets.
        cal_preds_loo:  (n_cal,) predictions for calibration point i from the
                        model trained WITHOUT i's fold (leave-one-out for
                        jackknife+, leave-fold-out for CV+).
        test_preds_loo: (n_cal, n_test) predictions for each test point from
                        each fold model (row i = model that omitted i's fold).
        alpha:          miscoverage level.

    Returns (lo, hi) arrays of shape (n_test,) with the distribution-free
    1 - 2*alpha marginal guarantee (requires exchangeability and the actual
    per-fold retrained models).
    """
    cal_y = np.asarray(cal_y, dtype=np.float64)
    cal_preds_loo = np.asarray(cal_preds_loo, dtype=np.float64)
    test_preds_loo = np.asarray(test_preds_loo, dtype=np.float64)
    if test_preds_loo.ndim == 1:
        test_preds_loo = test_preds_loo.reshape(-1, 1) if test_preds_loo.size == cal_y.size else test_preds_loo
    residuals = np.abs(cal_y - cal_preds_loo)                       # (n_cal,)
    lo_mat = test_preds_loo - residuals[:, None]                    # (n_cal, n_test)
    hi_mat = test_preds_loo + residuals[:, None]
    n = cal_y.size
    k_lo = int(np.floor((n + 1) * alpha))
    k_hi = int(np.ceil((n + 1) * (1.0 - alpha)))
    k_lo = min(max(k_lo, 1), n)
    k_hi = min(max(k_hi, 1), n)
    lo = np.sort(lo_mat, axis=0)[k_lo - 1]
    hi = np.sort(hi_mat, axis=0)[k_hi - 1]
    return lo, hi


@dataclass
class FrozenCVPlus:
    """Frozen-checkpoint CV+-flavored intervals with width-stability stats.

    See module docstring: with a single frozen model the construction reduces
    to split conformal for the interval itself; the K folds and the bootstrap
    are used to quantify how stable the conformal quantile (interval width)
    is under resampling of the calibration scenarios.
    """

    alpha: float = 0.1
    k_folds: int = 5
    n_bootstrap: int = 200
    seed: int = 42
    fold_quantiles_: np.ndarray = field(default_factory=lambda: np.array([]))
    bootstrap_quantiles_: np.ndarray = field(default_factory=lambda: np.array([]))
    q_pooled_: float | None = None

    def fit(self, cal_y: np.ndarray, cal_p: np.ndarray) -> "FrozenCVPlus":
        residuals = np.abs(np.asarray(cal_y, dtype=np.float64) - np.asarray(cal_p, dtype=np.float64))
        n = residuals.size
        self.q_pooled_ = conformal_quantile(residuals, self.alpha)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        folds = np.array_split(perm, min(self.k_folds, n))
        self.fold_quantiles_ = np.array(
            [conformal_quantile(residuals[f], self.alpha) for f in folds if f.size], dtype=np.float64
        )
        boots = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boots.append(conformal_quantile(residuals[idx], self.alpha))
        self.bootstrap_quantiles_ = np.asarray(boots, dtype=np.float64)
        return self

    def interval(self, test_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_pooled_ is None:
            raise RuntimeError("Call fit() first.")
        test_p = np.asarray(test_p, dtype=np.float64)
        return test_p - self.q_pooled_, test_p + self.q_pooled_

    def coverage(self, test_y: np.ndarray, test_p: np.ndarray) -> float:
        lo, hi = self.interval(test_p)
        test_y = np.asarray(test_y, dtype=np.float64)
        return float(np.mean((test_y >= lo) & (test_y <= hi)))

    def stability_report(self) -> dict:
        """Fold/bootstrap spread of the conformal quantile (width stability)."""
        if self.q_pooled_ is None:
            raise RuntimeError("Call fit() first.")
        b = self.bootstrap_quantiles_
        f = self.fold_quantiles_
        return {
            "q_pooled": float(self.q_pooled_),
            "k_folds": int(self.k_folds),
            "fold_quantiles": f.tolist(),
            "fold_q_std": float(np.std(f)) if f.size else None,
            "bootstrap_q_mean": float(np.mean(b)) if b.size else None,
            "bootstrap_q_std": float(np.std(b)) if b.size else None,
            "bootstrap_q_p05": float(np.percentile(b, 5)) if b.size else None,
            "bootstrap_q_p95": float(np.percentile(b, 95)) if b.size else None,
        }


def median_relative_width(test_p: np.ndarray, half_width: float, eps: float = 1e-30) -> float:
    """Median full interval width relative to |prediction| for a +/-q band."""
    test_p = np.asarray(test_p, dtype=np.float64)
    return float(np.median(2.0 * half_width / np.maximum(np.abs(test_p), eps)))

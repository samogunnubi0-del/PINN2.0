"""Split conformal prediction for regression (replaces MC Dropout UQ).

Implements the standard split conformal interval from Romano et al. (2019):
  nonconformity s_i = |y_i - ŷ_i|
  q = k-th order statistic with k = ceil((n+1)(1-alpha))
  interval at x*: [ŷ - q, ŷ + q]

Supports absolute (physical units) and relative (|y-ŷ|/|y|) scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1-alpha) quantile for split conformal."""
    scores = np.asarray(scores, dtype=np.float64)
    n = int(scores.size)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def nonconformity_absolute(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))


def nonconformity_relative(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    denom = np.maximum(np.abs(np.asarray(y_true, dtype=np.float64)), eps)
    return np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) / denom


@dataclass
class SplitConformalRegressor:
    """Symmetric split conformal intervals for scalar regression."""

    alpha: float = 0.1
    score_mode: str = "absolute"  # "absolute" | "relative"
    q: float | None = None
    n_calibration: int = 0
    calibration_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformalRegressor":
        if self.score_mode == "relative":
            scores = nonconformity_relative(y_true, y_pred)
        else:
            scores = nonconformity_absolute(y_true, y_pred)
        self.calibration_scores = scores
        self.n_calibration = int(scores.size)
        self.q = conformal_quantile(scores, self.alpha)
        return self

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q is None:
            raise RuntimeError("Call fit() before interval().")
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if self.score_mode == "relative":
            # Multiplicative band: ŷ * (1 ± q) clipped at zero for inventories.
            lo = np.maximum(y_pred * (1.0 - self.q), 0.0)
            hi = y_pred * (1.0 + self.q)
            return lo, hi
        return y_pred - self.q, y_pred + self.q

    def contains(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.q is None:
            raise RuntimeError("Call fit() before contains().")
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if self.score_mode == "relative":
            scores = nonconformity_relative(y_true, y_pred)
        else:
            scores = nonconformity_absolute(y_true, y_pred)
        return scores <= self.q

    def coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(self.contains(y_true, y_pred)))

    def median_relative_width(self, y_pred: np.ndarray, eps: float = 1e-30) -> float:
        """Median full interval width relative to the point prediction.

        Relative mode: [ŷ(1-q), ŷ(1+q)] -> width/ŷ = 2q (constant).
        Absolute mode: [ŷ-q, ŷ+q] -> 2q/|ŷ| per point, median reported.
        """
        if self.q is None:
            raise RuntimeError("Call fit() before median_relative_width().")
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if self.score_mode == "relative":
            return float(2.0 * self.q)
        width_rel = 2.0 * self.q / np.maximum(np.abs(y_pred), eps)
        return float(np.median(width_rel))


@dataclass
class MultiSpeciesConformal:
    """Per-species split conformal regressors."""

    species: list[str]
    alpha: float = 0.1
    score_mode: str = "absolute"
    models: dict[str, SplitConformalRegressor] = field(default_factory=dict)

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "MultiSpeciesConformal":
        """y_true, y_pred: (n, n_species)."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        self.models = {}
        for i, name in enumerate(self.species):
            m = SplitConformalRegressor(alpha=self.alpha, score_mode=self.score_mode)
            m.fit(y_true[:, i], y_pred[:, i])
            self.models[name] = m
        return self

    def coverage_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return {
            name: self.models[name].coverage(y_true[:, i], y_pred[:, i])
            for i, name in enumerate(self.species)
        }

    def width_report(self, y_pred: np.ndarray) -> dict[str, float]:
        """Per-species median relative interval width (see SplitConformalRegressor)."""
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return {
            name: self.models[name].median_relative_width(y_pred[:, i])
            for i, name in enumerate(self.species)
        }

    def to_dict(self) -> dict:
        out: dict = {
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "score_mode": self.score_mode,
            "species": {},
        }
        for name, m in self.models.items():
            out["species"][name] = {
                "q": m.q,
                "n_calibration": m.n_calibration,
                "median_calibration_score": float(np.median(m.calibration_scores))
                if m.calibration_scores.size
                else None,
            }
        return out

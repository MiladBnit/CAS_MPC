"""Scoring of probabilistic forecasts.

CRPS of a Gaussian mixture F = sum_i w_i N(m_i, s_i^2) at an outcome y, in
closed form (Grimit et al., 2006):

    CRPS(F, y) = sum_i w_i A(y - m_i, s_i^2) - 1/2 sum_ij w_i w_j A(m_i - m_j, s_i^2 + s_j^2),
    A(m, v)    = m (2 Phi(m / sqrt v) - 1) + 2 sqrt(v) phi(m / sqrt v)    (A(m, 0) = |m|).

It equals E|X - y| - 1/2 E|X - X'| (X, X' independent draws from F), has the
units of y, rewards forecasts that are both accurate and sharp, and is the
mean absolute error for a point forecast (all s_i = 0).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _A(m, v):
    s = np.sqrt(np.maximum(v, 0.0))
    safe = np.where(s > 0, s, 1.0)
    z = m / safe
    out = m * (2.0 * ndtr(z) - 1.0) + 2.0 * s * _INV_SQRT_2PI * np.exp(-0.5 * z * z)
    return np.where(s > 0, out, np.abs(m))


def crps_mixture(mean, std, w, y):
    """CRPS of the mixtures sum_i w[t, i] N(mean[t, i], std[t, i]^2) at y[t].
    mean, std, w of shape (T, N) (w normalized per row), y of shape (T,); returns (T,)."""
    mean, std, w, y = (np.asarray(a, float) for a in (mean, std, w, y))
    var = std ** 2
    t1 = np.sum(w * _A(y[:, None] - mean, var), axis=1)
    t2 = np.einsum("ti,tj,tij->t", w, w, _A(mean[:, :, None] - mean[:, None, :], var[:, :, None] + var[:, None, :]))
    return t1 - 0.5 * t2



def mixture_quantiles(mean, std, w, qs=(0.025, 0.975), iters=60):
    """Quantiles of the mixtures sum_i w[t, i] N(mean[t, i], std[t, i]^2) (shapes
    (T, N); a zero std is a point mass, so particles alone give their weighted
    empirical quantiles), by bisection on the mixture CDF. Returns (len(qs), T)."""
    mean, std, w = (np.asarray(a, float) for a in (mean, std, w))
    w = w / w.sum(axis=1, keepdims=True)
    safe = np.where(std > 0, std, 1.0)

    def cdf(x):
        d = x[:, None] - mean
        return np.sum(w * np.where(std > 0, ndtr(d / safe), (d >= 0).astype(float)), axis=1)
    out = []
    for q in qs:
        lo = (mean - 8 * std).min(axis=1) - 1e-12
        hi = (mean + 8 * std).max(axis=1) + 1e-12
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = cdf(mid) < q
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        out.append(hi)
    return np.array(out)

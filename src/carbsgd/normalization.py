"""Normalized coordinates (paper, Section 2.1, paragraph "Normalized coordinates").

    x_p = mu_x + Lambda_x x,   u_p = mu_u + Lambda_u u,   y_p = mu_y + Lambda_y y

with Lambda diagonal (stored as vectors). The constants are fixed: either given
(the bioreactor: centres and scales of the operating ranges, config bio.norm)
or computed from data (from_data: means and standard deviations of an input
sequence, the measured output and a state trajectory).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normalization:
    mu_x: np.ndarray       # (n_x,)
    Lambda_x: np.ndarray   # (n_x,)
    mu_u: np.ndarray       # (n_u,)
    Lambda_u: np.ndarray   # (n_u,)
    mu_y: np.ndarray       # (n_y,)
    Lambda_y: np.ndarray   # (n_y,)

    @staticmethod
    def from_data(u_p, y_p, x_p_baseline, std_floor: float) -> "Normalization":
        """u_p (T,) or (T, n_u), y_p (T,) or (T, n_y), x_p_baseline (T, n_x)."""
        def stats(a):
            a = np.asarray(a, dtype=float)
            a = a.reshape(len(a), -1)
            return a.mean(axis=0), np.maximum(a.std(axis=0), std_floor)
        mu_u, Lambda_u = stats(u_p)
        mu_y, Lambda_y = stats(y_p)
        mu_x, Lambda_x = stats(x_p_baseline)
        return Normalization(mu_x, Lambda_x, mu_u, Lambda_u, mu_y, Lambda_y)

    # physical -> normalized
    def x_norm(self, x_p):
        return (np.asarray(x_p) - self.mu_x) / self.Lambda_x

    def u_norm(self, u_p):
        return (np.asarray(u_p) - self.mu_u[0]) / self.Lambda_u[0]

    def y_norm(self, y_p):
        return (np.asarray(y_p) - self.mu_y[0]) / self.Lambda_y[0]

    # normalized -> physical
    def x_phys(self, x):
        return self.mu_x + self.Lambda_x * np.asarray(x)

    def u_phys(self, u):
        return self.mu_u[0] + self.Lambda_u[0] * np.asarray(u)

    def y_phys(self, y):
        return self.mu_y[0] + self.Lambda_y[0] * np.asarray(y)

    def to_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> "Normalization":
        return Normalization(**{k: np.asarray(v, dtype=float) for k, v in d.items()})

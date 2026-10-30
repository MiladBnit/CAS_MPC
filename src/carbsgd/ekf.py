"""EKF for one particle (paper, Section 2.2).

    prediction:  xhat_pred = f(xhat, u, theta),  F = df/dx,  P_pred = F P F^T + Q
    update:      H = dh/dx(xhat_pred),  S = H P_pred H^T + R,  K = P_pred H^T S^{-1}
                 xhat = xhat_pred + K (y - h(xhat_pred))
                 P = (I - K H) P_pred (I - K H)^T + K R K^T        (Joseph form)

The one-step log-likelihood is log N(y; h(xhat_pred), S)   (eq:marglik).

f(x, u, theta) and h(x) are plain functions; h may return a scalar (n_y = 1).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _hvec(h):
    return lambda x: jnp.atleast_1d(h(x))


def ekf_predict(f, xhat, P, u, theta, Q):
    x_pred = f(xhat, u, theta)
    F = jax.jacfwd(f, argnums=0)(xhat, u, theta)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def gaussian_loglik(e, S):
    """log N(e; 0, S) for a vector e."""
    n = e.shape[0]
    L = jnp.linalg.cholesky(S)
    z = jax.scipy.linalg.solve_triangular(L, e, lower=True)
    return -0.5 * (z @ z) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * n * jnp.log(2 * jnp.pi)


def innovation(h, x_pred, P_pred, y, R):
    """Innovation e = y - h(x_pred), its covariance S and H = dh/dx."""
    hv = _hvec(h)
    H = jax.jacfwd(hv)(x_pred)
    e = jnp.atleast_1d(y) - hv(x_pred)
    S = H @ P_pred @ H.T + R
    return e, S, H


def ekf_update(h, x_pred, P_pred, y, R):
    """Returns (xhat, P, loglik) with loglik = log N(y; h(x_pred), S)."""
    e, S, H = innovation(h, x_pred, P_pred, y, R)
    K = jnp.linalg.solve(S, H @ P_pred).T            # P_pred H^T S^{-1} (S, P symmetric)
    xhat = x_pred + K @ e
    I_KH = jnp.eye(P_pred.shape[0]) - K @ H
    P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    P = 0.5 * (P + P.T)
    return xhat, P, gaussian_loglik(e, S)


def ekf_step(f, h, xhat, P, u, y, theta, Q, R):
    """Prediction with (u_{t-1}, theta) followed by the update with y_t."""
    x_pred, P_pred = ekf_predict(f, xhat, P, u, theta, Q)
    return ekf_update(h, x_pred, P_pred, y, R)


def _finite_or(new, fallback):
    """new if all its entries are finite, else fallback (numerical guard)."""
    return jnp.where(jnp.all(jnp.isfinite(new)), new, fallback)


def augmented_ekf_step(f, h, xhat, theta, P_aug, u, y, Q, Q_theta, R):
    """EKF on the augmented state z = (x, theta), with theta a random walk:

        x_{t+1} = f(x_t, u_t, theta_t) + q_t,   theta_{t+1} = theta_t + w_t,  w_t ~ N(0, Q_theta I)
        y_t     = h(x_t) + r_t

    Prediction with (u_{t-1}, theta) and update with y_t. Returns the updated
    state estimate, the updated augmented covariance (n_x + n_theta square) and
    the parameter correction K_theta e (the EKF-guided score of the Stein
    filter). If the update is not finite, the prediction is kept.
    """
    n_x, n_th = xhat.shape[0], theta.shape[0]
    f_aug = lambda z: jnp.concatenate([f(z[:n_x], u, z[n_x:]), z[n_x:]])
    z = jnp.concatenate([xhat, theta])
    z_pred = f_aug(z)
    F = jax.jacfwd(f_aug)(z)
    Q_aug = jax.scipy.linalg.block_diag(Q, Q_theta * jnp.eye(n_th))
    P_pred = F @ P_aug @ F.T + Q_aug
    P_pred = 0.5 * (P_pred + P_pred.T)
    z_new, P_new, _ = ekf_update(lambda zz: h(zz[:n_x]), z_pred, P_pred, y, R)
    z_new = _finite_or(z_new, z_pred)
    P_new = _finite_or(P_new, P_pred)
    return z_new[:n_x], P_new, z_new[n_x:] - theta

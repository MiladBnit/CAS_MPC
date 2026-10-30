"""Score of the parameter posterior, eq:score.

    grad log pi_t(theta) = grad_theta log N(y_t; h(xhat_pred(theta)), S(theta))
                           - Sigma_{t-1}^{-1} (theta - thetahat_{t-1})

with xhat_pred(theta) = f(xhat_{t-1|t-1}, u_{t-1}, theta),
P_pred(theta) = F(theta) P_{t-1|t-1} F(theta)^T + Q, S(theta) = H P_pred H^T + R.
xhat_{t-1|t-1} and P_{t-1|t-1} are constants (stop-gradient).

The Gaussian prior is fitted to the particles at t-1 (Kotecha and Djuric,
2003). Since N < n_theta, Sigma_{t-1} = C + lambda_Sigma I, where C is the
sample covariance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .ekf import ekf_predict, ekf_update, gaussian_loglik, innovation


def loglik_theta(theta, f, h, xhat_prev, P_prev, u_prev, y, Q, R):
    """log p(y_t | theta, y_{1:t-1}) (eq:marglik) as a function of theta."""
    xhat_prev = jax.lax.stop_gradient(xhat_prev)
    P_prev = jax.lax.stop_gradient(P_prev)
    x_pred, P_pred = ekf_predict(f, xhat_prev, P_prev, u_prev, theta, Q)
    e, S, _ = innovation(h, x_pred, P_pred, y, R)
    return gaussian_loglik(e, S)


def loglik_score(theta, f, h, xhat_prev, P_prev, u_prev, y, Q, R):
    return jax.grad(loglik_theta)(theta, f, h, xhat_prev, P_prev, u_prev, y, Q, R)


def particle_prior(thetas, lambda_Sigma):
    """Gaussian fitted to the particles (N, n_theta): mean and Sigma^{-1}."""
    theta_hat = jnp.mean(thetas, axis=0)
    D = thetas - theta_hat
    C = D.T @ D / (thetas.shape[0] - 1)
    Sigma = C + lambda_Sigma * jnp.eye(thetas.shape[1])
    return theta_hat, jnp.linalg.inv(Sigma)


def isotropic_prior(theta_0, sigma_init):
    """Prior N(theta_0, sigma_init^2 I), used at t = 1."""
    n = theta_0.shape[0]
    return theta_0, jnp.eye(n) / sigma_init ** 2


def prior_score(theta, theta_hat, Sigma_inv):
    return -Sigma_inv @ (theta - theta_hat)


def score(theta, f, h, xhat_prev, P_prev, u_prev, y, Q, R, theta_hat, Sigma_inv):
    """eq:score for one particle."""
    return (loglik_score(theta, f, h, xhat_prev, P_prev, u_prev, y, Q, R)
            + prior_score(theta, theta_hat, Sigma_inv))


def likelihood_fisher(theta, f, h, xhat_prev, P_prev, u_prev, Q, R):
    """Exact Fisher information of the one-step likelihood eq:marglik at theta,
    F = J^T S^{-1} J with J = d h(f(xhat_{t-1|t-1}, u_{t-1}, theta)) / d theta and
    S the innovation covariance (Gauss-Newton curvature of -log p(y_t | theta);
    xhat_{t-1|t-1}, P_{t-1|t-1} fixed as in the score)."""
    hv = lambda x: jnp.atleast_1d(h(x))
    J = jax.jacrev(lambda th: hv(f(xhat_prev, u_prev, th)))(theta)          # (n_y, n_theta)
    x_pred, P_pred = ekf_predict(f, xhat_prev, P_prev, u_prev, theta, Q)
    H = jax.jacfwd(hv)(x_pred)
    S = H @ P_pred @ H.T + R
    return J.T @ jnp.linalg.solve(S, J)


def loglik_window(theta, f, h, x0, P0, U, Ys, Q, R):
    """Windowed version of eq:marglik: log p(y_t | theta, y_{1:t-1}) with the
    EKF run with theta over the last k steps, from x0 = xhat_{t-k|t-k},
    P0 = P_{t-k|t-k} (constants, stop-gradient): updates with y_{t-k+1..t-1},
    then the prediction of y_t. U = (u_{t-k}, ..., u_{t-1}), Ys = (y_{t-k+1},
    ..., y_t), both of length k. Only y_t is scored. k = 1 is loglik_theta.

    With k = 1 the conditional state does not depend on theta, so a parameter
    that reaches the measurement only after several steps (e.g. a residual on an
    unmeasured upstream state) has a zero score; the window keeps its effect
    on the conditional state for k steps."""
    x0, P0 = jax.lax.stop_gradient(x0), jax.lax.stop_gradient(P0)

    def step(carry, inp):
        x, P = carry
        u, y = inp
        x_pred, P_pred = ekf_predict(f, x, P, u, theta, Q)
        x_new, P_new, _ = ekf_update(h, x_pred, P_pred, y, R)
        return (x_new, P_new), None
    (x, P), _ = jax.lax.scan(step, (x0, P0), (U[:-1], Ys[:-1]))
    x_pred, P_pred = ekf_predict(f, x, P, U[-1], theta, Q)
    e, S, _ = innovation(h, x_pred, P_pred, Ys[-1], R)
    return gaussian_loglik(e, S)


def score_window(theta, f, h, x0, P0, U, Ys, Q, R, theta_hat, Sigma_inv):
    """eq:score with the windowed likelihood (window length = len(U))."""
    return (jax.grad(loglik_window)(theta, f, h, x0, P0, U, Ys, Q, R)
            + prior_score(theta, theta_hat, Sigma_inv))

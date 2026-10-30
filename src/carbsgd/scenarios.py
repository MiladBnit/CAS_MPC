"""Scenario propagation eq:scenario and margins eq:margins (used by the MPC)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .ekf import ekf_predict


def scenario_predict(model, thetas, xhat, P, U, Q, s0, unroll=1):
    """For every particle, EKF time updates under the plan U (L,).

    Returns mu (N, L, n_x), Sigma (N, L, n_x, n_x), margins m and spreads s with
    s^2 = a^T Sigma a + s0^2 (g affine, a = dg/dx): shape (N, L) for a scalar
    constraint (n_c = 1), (N, L, n_c) for a vector of constraints."""
    def one(theta, x0, P0):
        def step(carry, u):
            x, Pi = ekf_predict(model.f, carry[0], carry[1], u, theta, Q)
            return (x, Pi), (x, Pi)
        _, (mu, Sig) = jax.lax.scan(step, (x0, P0), U, unroll=unroll)
        return mu, Sig
    mu, Sig = jax.vmap(one)(thetas, xhat, P)
    gv = lambda x: jnp.atleast_1d(model.g(x))
    m = jax.vmap(jax.vmap(gv))(mu)                                  # (N, L, n_c)
    a = jax.jacfwd(gv)(mu[0, 0])                                    # (n_c, n_x), g affine
    s = jnp.sqrt(jnp.einsum("ci,nlij,cj->nlc", a, Sig, a) + s0 ** 2)
    if m.shape[-1] == 1:
        m, s = m[..., 0], s[..., 0]
    return mu, Sig, m, s

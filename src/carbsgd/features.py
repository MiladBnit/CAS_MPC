"""Feature map of the control-aware kernel (paper, Section 4.1).

    Ubar_t = (u*_{t|t-1}, ..., u*_{t+L-2|t-1}, u*_{t+L-2|t-1})            eq:shifted-plan
    xbar_{t+l+1|t}(theta) = f(xbar_{t+l|t}(theta), ubar_{t+l|t}, theta)   eq:feature-rollout
    Phi_t(theta) = col_{l=1..L_Phi} ( W_y^{1/2} h(xbar_{t+l|t}(theta)),
                                      g(xbar_{t+l|t}(theta)) )             eq:feature-map

h is the tracked output of the cost: model.track when the model has one (a
tracked quantity other than the measurement), else the measured output model.h.

xbar_{t|t} = mean_i xhat_{t|t}^(i) and Ubar_t are fixed during the SVGD
iterations; xbar_{t|t} is not differentiated. d_Phi = L_Phi (n_y + n_c) = 2 L_Phi.
The Jacobian J = dPhi_t/dtheta is computed by reverse-mode autodiff; the
forward sensitivity recursion eq:sensitivity is kept as a check (test 5).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def shifted_plan(U_prev):
    """eq:shifted-plan: drop the first input, repeat the last one."""
    return jnp.concatenate([U_prev[1:], U_prev[-1:]])


def feature_rollout(model, theta, xbar, Ubar, L_Phi):
    """States xbar_{t+l|t}(theta), l = 1..L_Phi (eq:feature-rollout)."""
    xbar = jax.lax.stop_gradient(xbar)

    def step(x, u):
        xn = model.f(x, u, theta)
        return xn, xn
    _, xs = jax.lax.scan(step, xbar, Ubar[:L_Phi])
    return xs


def Phi(model, theta, xbar, Ubar, L_Phi, W_y):
    """eq:feature-map, entries ordered (sqrt(W_y) h_1, g_1, sqrt(W_y) h_2, g_2, ...)."""
    xs = feature_rollout(model, theta, xbar, Ubar, L_Phi)
    hs = jax.vmap(getattr(model, "track", model.h))(xs)
    gs = jax.vmap(lambda x: jnp.atleast_1d(model.g(x)))(xs)          # (L_Phi, n_c)
    return jnp.concatenate([jnp.sqrt(W_y) * hs[:, None], gs], axis=1).ravel()


def Phi_and_J(model, theta, xbar, Ubar, L_Phi, W_y):
    """Phi_t(theta) and J = dPhi_t/dtheta (d_Phi, n_theta) by reverse mode."""
    fun = lambda th: Phi(model, th, xbar, Ubar, L_Phi, W_y)
    return fun(theta), jax.jacrev(fun)(theta)


def Phi_and_J_particles(model, thetas, xbar, Ubar, L_Phi, W_y):
    """Batched over particles: Phi (N, d_Phi), J (N, d_Phi, n_theta)."""
    return jax.vmap(lambda th: Phi_and_J(model, th, xbar, Ubar, L_Phi, W_y))(thetas)


def J_sensitivity(model, theta, xbar, Ubar, L_Phi, W_y):
    """J from the forward sensitivity recursion eq:sensitivity,
    Xi_{l+1} = A_l Xi_l + B_l, Xi_0 = 0, A = df/dx, B = df/dtheta."""
    dfdx = jax.jacfwd(model.f, argnums=0)
    dfdth = jax.jacfwd(model.f, argnums=2)
    dh = jax.grad(getattr(model, "track", model.h))
    dg = jax.jacfwd(lambda x: jnp.atleast_1d(model.g(x)))

    def step(carry, u):
        x, Xi = carry
        Xi_n = dfdx(x, u, theta) @ Xi + dfdth(x, u, theta)
        x_n = model.f(x, u, theta)
        rows = jnp.concatenate([(jnp.sqrt(W_y) * dh(x_n) @ Xi_n)[None], dg(x_n) @ Xi_n])
        return (x_n, Xi_n), rows
    Xi0 = jnp.zeros((xbar.shape[0], theta.shape[0]))
    _, rows = jax.lax.scan(step, (xbar, Xi0), Ubar[:L_Phi])
    return rows.reshape(-1, theta.shape[0])

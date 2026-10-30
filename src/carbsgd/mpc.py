"""Scenario SMPC on the particle posterior and CE-MPC (paper, Section 3).

Scenario i, parameters fixed along the horizon (eq:scenario):
    mu_{l+1} = f(mu_l, u_l, theta^(i)),  Sigma_{l+1} = A_l Sigma_l A_l^T + Q.
Margins (eq:margins), g(x) affine with a = dg/dx:
    m_l^(i) = g(mu_l^(i)),  (s_l^(i))^2 = a^T Sigma_l a + s0^2.
Chance constraint (eq:cc-mixture), imposed in the equivalent log form with a
slack sigma_l >= 0 (same feasible set; the gradient does not vanish deep in
the violated region, where F_N saturates):
    log sum_i w_i F_N(-m_l^(i) / s_l^(i)) >= log(1 - delta) - sigma_l.
Cost (eq:smpc) plus slack penalties:
    sum_i w_i sum_l W_y (h(mu_l^(i)) - r_{t+l})^2 + W_u sum_l (u_l - u_{l-1})^2
    + rho_1 sum_l sigma_l + rho_2 sum_l sigma_l^2,   u_{-1} := u_{t-1}.
CE-MPC ("ce"): one scenario, no covariance, g(mu_l) <= sigma_l.
Several constraints (model.g returning a vector of n_c affine margins, e.g. a
corridor X_min <= X <= X_max): one chance constraint per constraint and stage,
each with its own slack and its own delta (delta a scalar or a list of n_c).

The cost, the constraints and their Jacobians are evaluated with the JAX model
(compiled by XLA) and passed to IPOPT through CasADi callbacks.
"""

from __future__ import annotations

import time

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr, logsumexp, ndtr

from .scenarios import scenario_predict


class _Callback(ca.Callback):
    """y = fun(x) with Jacobian jac(x); x dense (n, 1), y dense (m, 1)."""

    def __init__(self, name, n, m, fun, jac=None, opts=None):
        ca.Callback.__init__(self)
        self.n, self.m, self.fun, self.jac = n, m, fun, jac
        self._jac_cb = None
        self.construct(name, opts or {})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.n, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.m, 1)

    def eval(self, arg):
        return [np.asarray(self.fun(np.asarray(arg[0]).ravel())).reshape(self.m, 1)]

    def has_jacobian(self):
        return self.jac is not None

    def get_jacobian(self, name, inames, onames, opts):
        self._jac_cb = _JacCallback(name, self.n, self.m, self.jac, opts)
        return self._jac_cb


class _JacCallback(ca.Callback):
    """Jacobian of a _Callback: inputs (x, nominal output), output dy/dx (m, n)."""

    def __init__(self, name, n, m, jac, opts=None):
        ca.Callback.__init__(self)
        self.n, self.m, self.jac = n, m, jac
        self.construct(name, opts or {})

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.n, 1) if i == 0 else ca.Sparsity.dense(self.m, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.m, self.n)

    def eval(self, arg):
        return [np.asarray(self.jac(np.asarray(arg[0]).ravel())).reshape(self.m, self.n)]


class ScenarioMPC:
    """solve() returns the optimal plan; prob() and logprob() evaluate the
    chance constraint of a given plan."""

    def __init__(self, model, *, N, L, W_y, W_u, delta, s0, Q, u_min, u_max, rho_1, rho_2,
                 mode="smpc", slack_tol=1e-6, ipopt=None, unroll=5):
        if mode not in ("smpc", "ce"):
            raise ValueError(mode)
        self.mode, self.N, self.L = mode, (1 if mode == "ce" else int(N)), int(L)
        self.n_c = n_c = int(getattr(model, "n_c", 1))
        self.delta = np.broadcast_to(np.asarray(delta, float), (n_c,)).copy()
        self.slack_tol = float(slack_tol)
        self.model = model
        L_ = self.L
        n_s = L_ * n_c                                         # slacks: one per stage and constraint
        Q = jnp.asarray(Q)
        Qs = jnp.zeros_like(Q) if mode == "ce" else Q

        def terms(z, prm):
            U, sig = z[:L_], z[L_:]
            mu, _, m, s = scenario_predict(model, prm["thetas"], prm["xhat"], prm["P"], U,
                                           Qs, s0, unroll)
            H = jax.vmap(jax.vmap(getattr(model, "track", model.h)))(mu)   # tracked output (N, L)
            track = prm["w"] @ jnp.sum((H - prm["r"][None, :]) ** 2, axis=1)
            du = U - jnp.concatenate([prm["u_prev"][None], U[:-1]])
            cost = (W_y * track + W_u * jnp.sum(du ** 2) + rho_1 * jnp.sum(sig)
                    + rho_2 * jnp.sum(sig ** 2))
            m3, s3 = m.reshape(m.shape[0], L_, n_c), s.reshape(s.shape[0], L_, n_c)
            sg = sig.reshape(L_, n_c)
            if mode == "smpc":
                logprob = logsumexp(log_ndtr(-m3 / s3), b=prm["w"][:, None, None], axis=0)
                g = (logprob + sg).ravel()
            else:
                g = (m3[0] - sg).ravel()
            return cost, g, m, s

        self._cost = jax.jit(lambda z, prm: terms(z, prm)[0])
        self._grad = jax.jit(jax.grad(lambda z, prm: terms(z, prm)[0]))
        self._g = jax.jit(lambda z, prm: terms(z, prm)[1])
        # g is affine in the slacks (dg/dsigma = +I for "smpc", -I for "ce"), so
        # only dg/dU is computed by forward-mode autodiff
        sign = 1.0 if mode == "smpc" else -1.0

        def jac_g(z, prm):
            JU = jax.jacfwd(lambda U: terms(jnp.concatenate([U, z[L_:]]), prm)[1])(z[:L_])
            return jnp.concatenate([JU, sign * jnp.eye(n_s)], axis=1)
        self._jac_g = jax.jit(jac_g)
        self._ms = jax.jit(lambda z, prm: terms(z, prm)[2:])
        self._predict = jax.jit(lambda U, prm: scenario_predict(
            model, prm["thetas"], prm["xhat"], prm["P"], U, Qs, s0, unroll)[:2])
        self._prm = None

        # IPOPT asks for f and g at the same point, and for grad f and jac g at
        # the same point: each pair is evaluated by one compiled function, cached
        # on the point, so the scenario propagation is done once per pair
        self._fg = jax.jit(lambda z, prm: terms(z, prm)[:2])
        self._dfg = jax.jit(lambda z, prm: (jax.grad(lambda zz: terms(zz, prm)[0])(z),
                                            jac_g(z, prm)))
        self._cache = {}

        n = L_ + n_s
        if mode == "smpc":
            self.lbg, self.ubg = np.tile(np.log(1.0 - self.delta), L_), np.full(n_s, np.inf)
        else:
            self.lbg, self.ubg = np.full(n_s, -np.inf), np.zeros(n_s)
        self.lbx = np.concatenate([np.full(L_, u_min), np.zeros(n_s)])
        self.ubx = np.concatenate([np.full(L_, u_max), np.full(n_s, np.inf)])
        self._f_cb = _Callback("smpc_f", n, 1, lambda z: self._eval("fg", z)[0],
                               lambda z: self._eval("dfg", z)[0][None, :])
        self._g_cb = _Callback("smpc_g", n, n_s, lambda z: self._eval("fg", z)[1],
                               lambda z: self._eval("dfg", z)[1])
        x = ca.MX.sym("x", n)
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
                "ipopt.hessian_approximation": "limited-memory", "ipopt.max_iter": 300,
                "ipopt.tol": 1e-8, "ipopt.acceptable_tol": 1e-6}
        opts.update({f"ipopt.{k}": v for k, v in (ipopt or {}).items()})
        self.solver = ca.nlpsol("smpc", "ipopt",
                                {"x": x, "f": self._f_cb(x), "g": self._g_cb(x)}, opts)

    def _eval(self, which, z):
        """Cached evaluation of the fused functions at the point z."""
        key = z.tobytes()
        hit = self._cache.get(which)
        if hit is None or hit[0] != key:
            fun = self._fg if which == "fg" else self._dfg
            hit = (key, [np.asarray(v) for v in fun(jnp.asarray(z), self._prm)])
            self._cache[which] = hit
        return hit[1]

    def _params(self, thetas, xhat, P, w, r, u_prev):
        return dict(thetas=jnp.asarray(thetas, float).reshape(self.N, -1),
                    xhat=jnp.asarray(xhat, float).reshape(self.N, self.model.n_x),
                    P=jnp.asarray(P, float).reshape(self.N, self.model.n_x, self.model.n_x),
                    w=jnp.asarray(w, float).ravel(), r=jnp.asarray(r, float).ravel(),
                    u_prev=jnp.asarray(float(u_prev)))

    def solve(self, thetas, xhat, P, w, r, u_prev, U_warm):
        self._prm = self._params(thetas, xhat, P, w, r, u_prev)
        self._cache = {}
        x0 = np.concatenate([np.clip(np.asarray(U_warm, float), self.lbx[:self.L],
                                     self.ubx[:self.L]), np.zeros(self.L * self.n_c)])
        t0 = time.perf_counter()
        sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        stats = self.solver.stats()
        z, cost = np.asarray(sol["x"]).ravel(), float(sol["f"])
        success, status, iters = (bool(stats["success"]), stats["return_status"],
                                  int(stats["iter_count"]))
        t_solve = time.perf_counter() - t0
        slack = z[self.L:]
        return dict(U=z[:self.L], slack=slack, slack_active=bool(slack.max() > self.slack_tol),
                    success=success, status=status, time=t_solve, cost=cost, iters=iters)

    def predict(self, U, thetas, xhat, P):
        """Scenario means (N, L, n_x) and covariances (N, L, n_x, n_x) of
        eq:scenario under the plan U, as used by the controller (normalized)."""
        prm = self._params(thetas, xhat, P, np.ones(self.N) / self.N, np.zeros(self.L), 0.0)
        mu, Sig = self._predict(jnp.asarray(U, float), prm)
        return np.asarray(mu), np.asarray(Sig)

    def logprob(self, U, thetas, xhat, P, w):
        """Left-hand side of the log-form chance constraint (without slack)."""
        prm = self._params(thetas, xhat, P, w, np.zeros(self.L), 0.0)
        z = jnp.concatenate([jnp.asarray(U, float), jnp.zeros(self.L * self.n_c)])
        g = np.asarray(self._g(z, prm))
        return g if self.n_c == 1 else g.reshape(self.L, self.n_c)

    def prob(self, U, thetas, xhat, P, w):
        """eq:cc-mixture left-hand side: (L,), or (L, n_c) for several constraints."""
        prm = self._params(thetas, xhat, P, w, np.zeros(self.L), 0.0)
        z = jnp.concatenate([jnp.asarray(U, float), jnp.zeros(self.L * self.n_c)])
        m, s = self._ms(z, prm)
        return np.asarray(jnp.tensordot(prm["w"], ndtr(-m / s), axes=1))

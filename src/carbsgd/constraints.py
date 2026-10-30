"""Parameter constraints theta in C for the constrained Stein filter (cRBSGD,
ADMM splitting of the SVGD update, as in the RBSVGD paper, Section 6).

C is the intersection of an affine set {A theta = b} and a box on a set of
coordinates that the affine rows do not touch, so the Euclidean projection is
closed form:

    Pi_C(v) = clip_box( v - A^T (A A^T)^{-1} (A v - b) ).

Per particle i and SVGD iteration, with the auxiliary z^(i) (feasible) and the
scaled dual nu^(i), both warm-started across time steps (RBSVGD paper, eq. (33)):

    score^(i) <- score^(i) - rho (theta^(i) - z^(i) + nu^(i))      (penalty, eq. (34))
    theta^(i) <- theta^(i) + epsilon phi(theta^(i))                 (SVGD step)
    z^(i)     <- Pi_C(theta^(i) + nu^(i))
    nu^(i)    <- nu^(i) + theta^(i) - z^(i)

and z is returned as the particle set (feasible after every step).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class Constraint:
    """Affine equalities A theta = b and a box lo <= theta[idx] <= hi."""

    def __init__(self, n_theta, A=None, b=None, box_idx=(), lo=(), hi=()):
        self.n_theta = int(n_theta)
        self.A = None if A is None else np.atleast_2d(np.asarray(A, float))
        self.b = None if b is None else np.atleast_1d(np.asarray(b, float))
        self.box_idx = np.asarray(box_idx, int)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        if self.A is not None:
            if self.A.shape != (len(self.b), self.n_theta):
                raise ValueError("A must be (m, n_theta) and b (m,)")
            if len(self.box_idx) and np.any(self.A[:, self.box_idx] != 0):
                raise ValueError("the box coordinates must not appear in the affine rows")
            self._AAt_inv = np.linalg.inv(self.A @ self.A.T)

    def project(self, thetas):
        """Pi_C for thetas (N, n_theta) or (n_theta,) (JAX)."""
        v = jnp.asarray(thetas)
        if self.A is not None:
            A, b, M = jnp.asarray(self.A), jnp.asarray(self.b), jnp.asarray(self._AAt_inv)
            v = v - ((v @ A.T - b) @ M) @ A
        if len(self.box_idx):
            idx = jnp.asarray(self.box_idx)
            v = v.at[..., idx].set(jnp.clip(v[..., idx], self.lo, self.hi))
        return v

    def clip_box(self, thetas):
        """The box part only (what a particle filter can apply after its proposal)."""
        v = jnp.asarray(thetas)
        if len(self.box_idx):
            idx = jnp.asarray(self.box_idx)
            v = v.at[..., idx].set(jnp.clip(v[..., idx], self.lo, self.hi))
        return v

    def residual(self, thetas):
        """A theta - b (N, m), the violation of the equalities (NumPy)."""
        if self.A is None:
            return np.zeros((np.atleast_2d(thetas).shape[0], 0))
        return np.atleast_2d(np.asarray(thetas)) @ self.A.T - self.b



class EqualityConstraint:
    """Nonlinear equalities c(theta) = 0, c: (n_theta,) -> (m,) (JAX), enforced
    by the modified differential method of multipliers (MDMM, Platt and Barr,
    1988) inside the SVGD update. Per particle i, with multipliers lambda^(i)
    kept across time steps:

        theta^(i)  <- theta^(i) + epsilon [phi(theta^(i)) - J_c(theta^(i))^T (lambda^(i) + rho c(theta^(i)))]
        lambda^(i) <- lambda^(i) + epsilon eta c(theta^(i))

    i.e. descent on the augmented Lagrangian of -log p in theta and ascent in
    lambda; no projection is needed, feasibility is reached asymptotically. The
    constraint force acts on each particle directly, not through the kernel
    average of the SVGD direction phi (in the score it would be smeared over
    neighbouring particles, strongly so with the control-aware kernel).
    With the Fisher-Adam step, the force and rho J^T J are added after the
    moving averages (no momentum; a Gauss-Newton step for the constraint)."""

    def __init__(self, c, m, rho=1.0, eta=1.0, inequality=False, c_aux=None, aux_fn=None):
        """inequality: c(theta) <= 0 instead of = 0 (augmented Lagrangian of
        Rockafellar: force J^T max(0, lambda + rho c), multipliers kept >= 0,
        lambda <- max(0, lambda + epsilon eta c)).

        Moving collocation points (optional): c_aux(theta, aux) is the full
        constraint vector (m entries) given points aux, which the filter
        computes each step as aux_fn(xbar, Ubar, theta_mean) (its state
        estimate, shifted plan and particle mean); c(theta) is then the part
        without them (used where no filter state exists, e.g. pre-training).
        The last points used are kept in last_aux."""
        self.c, self.m, self.rho, self.eta = c, int(m), float(rho), float(eta)
        self.inequality = bool(inequality)
        self.c_aux, self.aux_fn = c_aux, aux_fn
        self.last_aux = None

    def cc(self, theta, aux=None):
        """The constraint vector, with the moving points when given."""
        return self.c(theta) if (self.c_aux is None or aux is None) else self.c_aux(theta, aux)

    def violation(self, thetas, aux=None):
        """Largest violation per particle: max |c| (equalities) or max(0, max c)."""
        import jax
        v = jax.vmap(lambda th: self.cc(th, aux))(jnp.asarray(thetas))
        return (jnp.maximum(v.max(axis=1), 0.0) if self.inequality
                else jnp.abs(v).max(axis=1))

    def newton_project(self, thetas, iters=50, damping=1e-3):
        """Damped minimum-norm Gauss-Newton (Levenberg-Marquardt) steps onto
        c = 0 (used for the initial particles):
        theta <- theta - J^T (J J^T + damping I)^{-1} c(theta), a step that is
        not finite or does not reduce |c| is rejected and the damping raised."""
        import jax

        def one(th):
            mu = damping
            for _ in range(iters):
                J = jax.jacrev(self.c)(th)
                cv = self.c(th)
                new = th - J.T @ jnp.linalg.solve(J @ J.T + mu * jnp.eye(self.m), cv)
                ok = jnp.all(jnp.isfinite(new)) & (jnp.sum(self.c(new) ** 2) < jnp.sum(cv ** 2))
                th = jnp.where(ok, new, th)
                mu = jnp.where(ok, mu * 0.3, mu * 10.0)
            return th
        return jax.jit(jax.vmap(one))(jnp.asarray(thetas))

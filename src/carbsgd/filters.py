"""Rao-Blackwellized Stein filter for (x_t, theta) (paper, Section 2.2).

RBSGD with kernel variant "param" (k_theta) or "ca" (k_omega), and one of two
scores:

score "marginal" (paper, eq:score). At each time t:
  1. EKF step of every particle with theta_{t-1}^(i)          -> xhat_{t|t}, P_{t|t}
  2. score eq:score at the SVGD iterate, with xhat_{t-1|t-1}, P_{t-1|t-1} fixed
     and the Gaussian prior fitted to the particles at t-1 (isotropic prior at
     t = 1); prior "none" drops the prior term (likelihood score only). With
     score_window k > 1 the likelihood of y_t is computed by running each
     particle's EKF with the iterate over the last k steps from its stored
     xhat_{t-k|t-k} (score.loglik_window; k = 1 is eq:score)
  3. M SVGD iterations eq:svgd with the chosen kernel.

score "ekf" (EKF-guided, as in the RBSVGD code). Every
particle carries an EKF on the augmented state (x, theta), theta a random walk
with variance Q_theta (ekf.augmented_ekf_step). At each time t:
  1. augmented EKF step of every particle with theta_{t-1}^(i) -> xhat_{t|t},
     the augmented covariance, and the score s^(i) = K_theta e (the parameter
     correction of the Kalman update, a Gauss-Newton step scaled by the
     parameter covariance; the past measurements enter through it, so no
     separate prior term is used);
  2. one SVGD step from theta_{t-1}^(i) with these scores (epsilon ~ 1 takes a
     full Kalman-sized step).

score "ekf_svgd" (the two-stage RBEKFSVGD filter of the RBSVGD paper). At each
time t:
  0. augmented EKF step of every particle with theta_{t-1}^(i) (as for "ekf"):
     xhat_{t|t}, the augmented covariance, the correction K_theta e;
  1. EKF warm start: theta <- theta_{t-1} + lr_ekf (1/N) sum_j k(theta_j, theta_i)
     (K_theta e)_j (Euclidean kernel, no repulsion: alpha_ekf = 0);
  2. M SVGD iterations from there with the paper's kernel ("param" or "ca") and
     the score of the one-step likelihood (from xhat_{t-1|t-1}, as eq:score)
     plus a per-particle recursive prior N(theta_{t-1}^(i), P_theta^(i) + Q_theta I)
     (P_theta^(i) the parameter block of the particle's augmented covariance);
     the Fisher step uses the exact likelihood Fisher plus that prior's
     precision; the constraints (ADMM, MDMM) act here;
  3. the parameter blocks of the augmented covariances are inflated to at least
     the empirical covariance of the particles.

For "ca" the feature map uses xbar_{t|t} = mean_i xhat_{t|t}^(i) and the
shifted plan Ubar_t. The step is the plain step eq:svgd-step (optimizer "sgd"),
Adam on the SVGD direction ("adam"), or ("fisher", score "marginal" only) a
natural-gradient step (Fisher-Adam): with
exponential moving averages m of the SVGD direction phi (beta1) and F of the
curvature F_i + Sigma^{-1} (beta2; F_i the exact Fisher of the one-step
likelihood, score.likelihood_fisher, Sigma^{-1} the prior precision),
    theta_i <- theta_i + epsilon (F_hat_i + d I)^{-1} m_hat_i
(bias-corrected averages, damping d, optional cap on the norm of the solve);
or ("svn") the Stein variational Newton step (Detommaso et al., 2018, block-
diagonal form), with the curvature averaged by the kernel like the direction:
    H_i = (1/N) sum_j [ k_ji^2 (F_j + Sigma^{-1}) + alpha grad_j k_ji grad_j k_ji^T ],
    theta_i <- theta_i + epsilon (H_i + d I)^{-1} phi_i,
no moving averages. The kernel mass cancels between phi_i and H_i: an isolated
particle takes the Newton (Gauss-Newton) step of its own posterior, and epsilon = 1
is the full Bayesian update for a model linear in theta; with "fisher" the
step shrinks by the kernel mass sum_j k_ji / N. alpha weights the repulsion term and
scale_h the median-heuristic bandwidth of k_theta, scale_h_phi that of k_Phi
(None: scale_h; all 1 in the paper).
Output: equally weighted {theta_t^(i), xhat_{t|t}^(i), P_{t|t}^(i)}   eq:rb-post

RBPF (baseline): random-walk proposal theta_t = theta_{t-1}
+ eta, eta ~ N(0, sigma_rw^2 I), one EKF per particle run with theta_t, weights
multiplied by the one-step likelihood eq:marglik, systematic resampling when
the effective sample size falls below ess_frac * N (the EKF mean and
covariance are copied with the resampled particle).

Constrained variant (cRBSGD, score "marginal"): with a constraints.Constraint
C, the ADMM splitting of the RBSVGD paper (Section 6): every SVGD
iteration adds the penalty -rho (theta - z + nu) to the scores, then
z <- Pi_C(theta + nu), nu <- nu + theta - z; z (feasible) is returned as the
particle set and nu is kept across time steps. RBPF with a constraint applies
its box part after the proposal (clipping); it has no counterpart for the
equalities.

Equality constraints c(theta) = 0 (constraints.EqualityConstraint, argument
eq_constraint): MDMM inside the SVGD update, per particle multipliers kept
across time steps; RBPF has no counterpart.

sigma_init may be a vector (one standard deviation per parameter).

The model object must provide f(x, u, theta), h(x) and, for "ca", g(x).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .ekf import _finite_or, augmented_ekf_step, ekf_step
from .features import Phi_and_J_particles
from .kernels import VARIANTS, gaussian_kernel, kernel_and_grad, svgd_direction
from .score import isotropic_prior, likelihood_fisher, particle_prior, score
from .score import score_window as _score_window

SCORES = ("marginal", "ekf", "ekf_svgd")


def _block(tree):
    return jax.block_until_ready(tree)


@dataclass
class ParticleSet:
    """{theta^(i), xhat^(i), P^(i)} with equal weights w^(i) = 1/N. For the
    "ekf" score, P_aug holds the augmented (x, theta) covariances; P is always
    the state covariance used by the controller."""
    thetas: jnp.ndarray            # (N, n_theta)
    xhat: jnp.ndarray              # (N, n_x)
    P: jnp.ndarray                 # (N, n_x, n_x)
    w: jnp.ndarray                 # (N,)
    P_aug: jnp.ndarray | None = None   # (N, n_x + n_theta, n_x + n_theta)


def initial_particles(key, theta_0, sigma_init, N, x0, P0):
    """theta^(i) ~ N(theta_0, sigma_init^2 I); common initial state estimate.
    theta_0 of shape (n_ens, n_theta) (an ensemble of pre-trained networks):
    theta^(i) = member i mod n_ens, plus N(0, sigma_init^2 I)."""
    theta_0 = jnp.asarray(theta_0)
    if theta_0.ndim == 2:
        theta_0 = theta_0[jnp.arange(N) % theta_0.shape[0]]
    thetas = theta_0 + sigma_init * jax.random.normal(key, (N, theta_0.shape[-1]))
    return ParticleSet(thetas, jnp.tile(x0, (N, 1)), jnp.tile(P0, (N, 1, 1)),
                       jnp.full(N, 1.0 / N))


def ekf_particles(model, thetas, xhat, P, u, y, Q, R):
    """One EKF step per particle; returns xhat, P and the log-likelihoods
    (the previous estimate is kept for a particle whose update is not finite)."""
    def step(th, x, Pi):
        xn, Pn, ll = ekf_step(model.f, model.h, x, Pi, u, y, th, Q, R)
        return _finite_or(xn, x), _finite_or(Pn, Pi), ll
    return jax.vmap(step)(thetas, xhat, P)


class RBSGD:
    def __init__(self, model, variant, Q, R, *, M, epsilon, omega, lambda_Sigma,
                 theta_0, sigma_init, L_Phi=None, W_y=1.0, h_floor=1e-8, optimizer="sgd",
                 score_mode="marginal", Q_theta=0.0, alpha=1.0, scale_h=1.0, scale_h_phi=None,
                 prior="particles",
                 fisher_beta1=0.9, fisher_beta2=0.999, fisher_damping=1e-3, fisher_max_step=None,
                 score_window=1, constraint=None, rho_admm=1.0, eq_constraint=None,
                 lr_ekf=1.0):
        if variant not in VARIANTS:
            raise ValueError(variant)
        if optimizer not in ("sgd", "adam", "fisher", "svn"):
            raise ValueError(optimizer)
        if optimizer in ("fisher", "svn") and score_mode == "ekf":
            raise ValueError("optimizer fisher or svn needs score marginal or ekf_svgd")
        if score_mode not in SCORES:
            raise ValueError(score_mode)
        if prior not in ("particles", "none"):
            raise ValueError(prior)
        self.prior_mode = prior
        self.constraint, self.rho_admm = constraint, float(rho_admm)
        if constraint is not None and score_mode == "ekf":
            raise ValueError("constraints need score marginal or ekf_svgd")
        self._nu = None                               # ADMM scaled duals (N, n_theta)
        self.eq = eq_constraint
        if eq_constraint is not None and score_mode == "ekf":
            raise ValueError("constraints need score marginal or ekf_svgd")
        self._lam = None                              # MDMM multipliers (N, m)
        self.window = int(score_window)
        if self.window < 1:
            raise ValueError("score_window must be >= 1")
        self._post = deque(maxlen=self.window)       # xhat_{s|s}, P_{s|s} of the last k times
        self._u = deque(maxlen=self.window)          # u_{t-k}, ..., u_{t-1}
        self._y = deque(maxlen=self.window)          # y_{t-k+1}, ..., y_t
        self.model, self.variant, self.score_mode = model, variant, score_mode
        self.Q, self.R = jnp.asarray(Q), jnp.asarray(R)
        self.M, self.epsilon, self.omega = int(M), float(epsilon), float(omega)
        self.lambda_Sigma = float(lambda_Sigma)
        self.sigma_init = jnp.asarray(sigma_init, float)   # scalar or (n_theta,)
        theta_0 = jnp.asarray(theta_0)       # (n_theta,), or an ensemble (n_ens, n_theta)
        self.prior0 = (particle_prior(theta_0, self.lambda_Sigma + self.sigma_init ** 2)
                       if theta_0.ndim == 2 else isotropic_prior(theta_0, self.sigma_init))
        self.L_Phi, self.W_y, self.h_floor = L_Phi, float(W_y), float(h_floor)
        self.alpha, self.scale_h = float(alpha), float(scale_h)
        self.scale_h_phi = None if scale_h_phi is None else float(scale_h_phi)
        self.t = 0
        self.optimizer = optimizer
        self._opt = optax.adam(self.epsilon) if optimizer == "adam" else None
        self._opt_state = None
        b1, b2, damp = float(fisher_beta1), float(fisher_beta2), float(fisher_damping)
        max_step = None if fisher_max_step is None else float(fisher_max_step)

        self._ekf = jax.jit(lambda th, x, P, u, y: ekf_particles(model, th, x, P, u, y,
                                                                 self.Q, self.R))
        self._prior = jax.jit(lambda th: particle_prior(th, self.lambda_Sigma))

        def scores(thetas, xprev, Pprev, u, y, theta_hat, Sigma_inv):
            s = lambda th, x, P: score(th, model.f, model.h, x, P, u, y, self.Q, self.R,
                                       theta_hat, Sigma_inv)
            return jax.vmap(s)(thetas, xprev, Pprev)
        self._scores = jax.jit(scores)

        def scores_w(thetas, X0, P0, U, Ys, theta_hat, Sigma_inv):
            s = lambda th, x, P: _score_window(th, model.f, model.h, x, P, U, Ys, self.Q, self.R,
                                              theta_hat, Sigma_inv)
            return jax.vmap(s)(thetas, X0, P0)
        self._scores_w = jax.jit(scores_w)        # compiled once per window length

        def fishers(thetas, xprev, Pprev, u, Sigma_inv):
            F = jax.vmap(lambda th, x, P: likelihood_fisher(th, model.f, model.h, x, P, u,
                                                            self.Q, self.R))(thetas, xprev, Pprev)
            return F + Sigma_inv
        self._fishers = jax.jit(fishers)

        def fisher_step(thetas, phi, Fn, state, force=None, JtJ=None):
            """Fisher-Adam step; a constraint force (and its curvature JtJ) is
            added after the moving averages, so it has no momentum and the
            constraint part is a Gauss-Newton step."""
            m, F, t = state
            t = t + 1
            m = b1 * m + (1 - b1) * phi
            F = b2 * F + (1 - b2) * Fn
            m_hat, F_hat = m / (1 - b1 ** t), F / (1 - b2 ** t)
            if force is not None:
                m_hat, F_hat = m_hat + force, F_hat + JtJ
            n = thetas.shape[1]
            raw = jax.vmap(lambda Fi, mi: jnp.linalg.solve(Fi + damp * jnp.eye(n), mi))(F_hat, m_hat)
            if max_step is not None:
                nrm = jnp.linalg.norm(raw, axis=1, keepdims=True)
                raw = raw * jnp.minimum(1.0, max_step / jnp.maximum(nrm, 1e-30))
            return thetas + self.epsilon * raw, (m, F, t)

        def svn_step(thetas, K, G, phi, Fn, force=None, JtJ=None):
            """Stein variational Newton step (see the module docstring). A constraint
            force and its Gauss-Newton curvature act per particle, scaled by the
            curvature mass c_i = (1/N) sum_j k_ji^2, so that they enter as for an
            isolated particle (as in fisher_step)."""
            N, n = thetas.shape
            H = (jnp.einsum("ji,jab->iab", K ** 2, Fn)
                 + self.alpha * jnp.einsum("jia,jib->iab", G, G)) / N
            if force is not None:
                c = jnp.sum(K ** 2, axis=0) / N
                phi = phi + c[:, None] * force
                H = H + c[:, None, None] * JtJ
            raw = jax.vmap(lambda Hi, pi: jnp.linalg.solve(Hi + damp * jnp.eye(n), pi))(H, phi)
            if max_step is not None:
                nrm = jnp.linalg.norm(raw, axis=1, keepdims=True)
                raw = raw * jnp.minimum(1.0, max_step / jnp.maximum(nrm, 1e-30))
            return thetas + self.epsilon * raw

        Q_th = float(Q_theta)
        self.Q_theta, self.lr_ekf = Q_th, float(lr_ekf)

        def scores_pp(thetas, xprev, Pprev, u, y, means, precs):
            """eq:score with a per-particle Gaussian prior (score ekf_svgd)."""
            s = lambda th, x, P, mu, Pi: score(th, model.f, model.h, x, P, u, y, self.Q, self.R,
                                               mu, Pi)
            return jax.vmap(s)(thetas, xprev, Pprev, means, precs)
        self._scores_pp = jax.jit(scores_pp)

        def fishers_pp(thetas, xprev, Pprev, u, precs):
            F = jax.vmap(lambda th, x, P: likelihood_fisher(th, model.f, model.h, x, P, u,
                                                            self.Q, self.R))(thetas, xprev, Pprev)
            return F + precs
        self._fishers_pp = jax.jit(fishers_pp)

        def stage1(thetas, dth):
            """EKF warm start: kernel-weighted average of the EKF corrections."""
            K, _ = gaussian_kernel(thetas, self.h_floor, self.scale_h)
            return thetas + self.lr_ekf * (K.T @ dth) / thetas.shape[0]
        self._stage1 = jax.jit(stage1)

        def inflate(P_aug, thetas):
            n_x = P_aug.shape[1] - thetas.shape[1]
            D = thetas - jnp.mean(thetas, axis=0)
            C = D.T @ D / max(thetas.shape[0] - 1, 1)
            blk = jnp.maximum(P_aug[:, n_x:, n_x:], C[None])
            return P_aug.at[:, n_x:, n_x:].set(blk)
        self._inflate = jax.jit(inflate)
        self._aug_ekf = jax.jit(lambda th, x, Pa, u, y: jax.vmap(
            lambda thi, xi, Pai: augmented_ekf_step(model.f, model.h, xi, thi, Pai, u, y,
                                                    self.Q, Q_th, self.R))(th, x, Pa))

        if variant == "ca":
            L = int(L_Phi)
            self._features = jax.jit(
                lambda th, xbar, Ubar: Phi_and_J_particles(model, th, xbar, Ubar, L, self.W_y))

        if constraint is not None:
            rho = self.rho_admm

            def admm(thetas, z, nu):
                z = constraint.project(thetas + nu)
                return z, nu + thetas - z
            self._admm = jax.jit(admm)
            self._penalty = jax.jit(lambda S, th, z, nu: S - rho * (th - z + nu))

        if eq_constraint is not None:
            eqc = eq_constraint

            ineq = getattr(eqc, "inequality", False)

            def mdmm_terms(thetas, lam, aux):
                cfun = lambda th: eqc.cc(th, aux)
                cv = jax.vmap(cfun)(thetas)                           # (N, m)
                J = jax.vmap(jax.jacrev(cfun))(thetas)                # (N, m, n)
                if ineq:                                              # active rows only
                    mult = jnp.maximum(lam + eqc.rho * cv, 0.0)
                    act = (mult > 0).astype(J.dtype)
                    pen = jnp.einsum("imn,im->in", J, mult)
                    return pen, eqc.rho * jnp.einsum("imn,im,imk->ink", J, act, J)
                pen = jnp.einsum("imn,im->in", J, lam + eqc.rho * cv)
                return pen, eqc.rho * jnp.einsum("imn,imk->ink", J, J)
            self._mdmm_terms = jax.jit(mdmm_terms)
            # the multiplier ODE d lambda / d tau = eta c(theta) discretized with the
            # same step epsilon as the SVGD update
            eps_lam = self.epsilon * eqc.eta
            cvals = lambda thetas, aux: jax.vmap(lambda th: eqc.cc(th, aux))(thetas)
            if ineq:
                self._mdmm_lam = jax.jit(lambda thetas, lam, aux: jnp.maximum(
                    lam + eps_lam * cvals(thetas, aux), 0.0))
            else:
                self._mdmm_lam = jax.jit(lambda thetas, lam, aux: lam + eps_lam * cvals(thetas, aux))
            # moving collocation points: the state estimate and the end of the
            # prediction under the shifted plan, with the particle mean
            self._aux = jax.jit(eqc.aux_fn) if getattr(eqc, "aux_fn", None) is not None else None

        def update(thetas, S, Phis, Js, opt_state, Fn, force, JtJ):
            K, G, info = kernel_and_grad(variant, thetas, Phis, Js, self.omega, self.h_floor,
                                         self.scale_h, self.scale_h_phi)
            phi = svgd_direction(K, G, S, self.alpha)                # eq:svgd-dir
            if self.optimizer == "svn":
                return _finite_or(svn_step(thetas, K, G, phi, Fn, force, JtJ), thetas), opt_state, info
            if self.optimizer == "fisher":
                new, opt_state = fisher_step(thetas, phi, Fn, opt_state, force, JtJ)
                return _finite_or(new, thetas), opt_state, info
            if force is not None:     # per-particle constraint force, outside the kernel average
                phi = phi + force
            if self._opt is None:
                new = thetas + self.epsilon * phi                     # eq:svgd-step
            else:
                step, opt_state = self._opt.update(-phi, opt_state)   # ascent along phi
                new = thetas + step
            return _finite_or(new, thetas), opt_state, info
        self._update = jax.jit(update)

    def reset(self, seed=None):
        """Back to t = 0 (isotropic prior and fresh optimizer state), keeping
        the compiled functions (seed: unused, for the same interface as RBPF)."""
        self.t = 0
        self._opt_state = None
        self._nu = None
        self._lam = None
        self._post.clear()
        self._u.clear()
        self._y.clear()

    def _svgd(self, thetas, S, xbar, Ubar, Fn=None, force=None, JtJ=None):
        Phis = Js = None
        if self.variant == "ca":
            Phis, Js = self._features(thetas, xbar, jnp.asarray(Ubar))
        thetas, self._opt_state, info = _block(
            self._update(thetas, S, Phis, Js, self._opt_state, Fn, force, JtJ))
        return thetas, info

    def step(self, ps: ParticleSet, u_prev, y, Ubar=None):
        """Filter step at time t with u_{t-1} and y_t. Returns the new particle
        set, the timings (s) of the filter and kernel parts, and the bandwidths."""
        tim = {"filter": 0.0, "kernel": 0.0}
        thetas, info = ps.thetas, {}
        if self._opt is not None and self._opt_state is None:
            self._opt_state = self._opt.init(thetas)
        if self.optimizer == "fisher" and self._opt_state is None:
            N, n = thetas.shape
            self._opt_state = (jnp.zeros((N, n)), jnp.zeros((N, n, n)), jnp.asarray(0.0))

        if self.score_mode == "ekf_svgd":
            return self._step_ekf_svgd(ps, u_prev, y, Ubar, tim)

        if self.score_mode == "ekf":
            t0 = time.perf_counter()
            P_aug = ps.P_aug
            if P_aug is None:     # initial augmented covariance: blockdiag(P0, sigma_init^2 I)
                n_th = thetas.shape[1]
                P_aug = jax.vmap(lambda P: jax.scipy.linalg.block_diag(
                    P, self.sigma_init ** 2 * jnp.eye(n_th)))(ps.P)
            xhat, P_aug, S = _block(self._aug_ekf(thetas, ps.xhat, P_aug, u_prev, y))
            n_x = xhat.shape[1]
            P = P_aug[:, :n_x, :n_x]
            tim["filter"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            thetas, info = self._svgd(thetas, S, jnp.mean(xhat, axis=0), Ubar)
            tim["kernel"] += time.perf_counter() - t0
            self.t += 1
            return (ParticleSet(thetas, xhat, P, ps.w, P_aug), tim,
                    {k: float(v) for k, v in info.items()})

        t0 = time.perf_counter()
        if self.window > 1:                           # history for the windowed score
            if not self._post:
                self._post.append((ps.xhat, ps.P))
            self._u.append(float(u_prev))
            self._y.append(jnp.asarray(y, float).reshape(()))
            w = min(len(self._post), len(self._u))
            X0, P0 = self._post[-w]                   # xhat_{t-w|t-w}, P_{t-w|t-w}
            U = jnp.asarray(list(self._u)[-w:])
            Ys = jnp.stack(list(self._y)[-w:])
        xhat, P, _ = _block(self._ekf(ps.thetas, ps.xhat, ps.P, u_prev, y))
        if self.prior_mode == "none":   # zero precision: likelihood score only
            prior = (self.prior0[0], jnp.zeros_like(self.prior0[1]))
        else:
            prior = self.prior0 if self.t == 0 else _block(self._prior(ps.thetas))
        tim["filter"] += time.perf_counter() - t0
        xbar = jnp.mean(xhat, axis=0)
        z = thetas                                    # ADMM: the particles are the feasible z
        if self.constraint is not None and self._nu is None:
            self._nu = jnp.zeros_like(thetas)
        if self.eq is not None and self._lam is None:
            self._lam = jnp.zeros((thetas.shape[0], self.eq.m))
        aux = None
        if self.eq is not None and self._aux is not None and Ubar is not None:
            aux = self._aux(xbar, jnp.asarray(Ubar), jnp.mean(thetas, axis=0))
            self.eq.last_aux = aux
        for _ in range(self.M):
            t0 = time.perf_counter()
            if self.window > 1:
                S = _block(self._scores_w(thetas, X0, P0, U, Ys, *prior))
            else:
                S = _block(self._scores(thetas, ps.xhat, ps.P, u_prev, y, *prior))
            Fn = (_block(self._fishers(thetas, ps.xhat, ps.P, u_prev, prior[1]))
                  if self.optimizer in ("fisher", "svn") else None)
            if self.constraint is not None:
                S = self._penalty(S, thetas, z, self._nu)
            force = JtJ = None
            if self.eq is not None:
                pen, JtJ = self._mdmm_terms(thetas, self._lam, aux)
                force = -pen
            tim["filter"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            thetas, info = self._svgd(thetas, S, xbar, Ubar, Fn, force, JtJ)
            if self.constraint is not None:
                z, self._nu = _block(self._admm(thetas, z, self._nu))
            if self.eq is not None:
                self._lam = _block(self._mdmm_lam(thetas, self._lam, aux))
            tim["kernel"] += time.perf_counter() - t0
        if self.constraint is not None:
            thetas = z
        self.t += 1
        if self.window > 1:
            self._post.append((xhat, P))
        return ParticleSet(thetas, xhat, P, ps.w), tim, {k: float(v) for k, v in info.items()}

    def _step_ekf_svgd(self, ps, u_prev, y, Ubar, tim):
        """score ekf_svgd: EKF warm start (stages 0-1) and SVGD with a
        per-particle recursive prior (stage 2); see the module docstring."""
        thetas = ps.thetas
        N, n_th = thetas.shape
        t0 = time.perf_counter()
        P_aug = ps.P_aug if ps.P_aug is not None else _initial_P_aug(ps.P, self.sigma_init, n_th)
        n_x = ps.xhat.shape[1]
        xhat, P_aug_new, dth = _block(self._aug_ekf(thetas, ps.xhat, P_aug, u_prev, y))
        P_th_pred = P_aug[:, n_x:, n_x:] + self.Q_theta * jnp.eye(n_th)
        precs = jnp.linalg.inv(P_th_pred)
        th_prev = thetas
        thetas = self._stage1(thetas, dth)
        tim["filter"] += time.perf_counter() - t0
        xbar = jnp.mean(xhat, axis=0)
        z = ps.thetas                                 # ADMM: the particles are the feasible z
        if self.constraint is not None and self._nu is None:
            self._nu = jnp.zeros_like(thetas)
        if self.eq is not None and self._lam is None:
            self._lam = jnp.zeros((N, self.eq.m))
        aux = None
        if self.eq is not None and self._aux is not None and Ubar is not None:
            aux = self._aux(xbar, jnp.asarray(Ubar), jnp.mean(thetas, axis=0))
            self.eq.last_aux = aux
        info = {}
        for _ in range(self.M):
            t0 = time.perf_counter()
            S = _block(self._scores_pp(thetas, ps.xhat, ps.P, u_prev, y, th_prev, precs))
            Fn = (_block(self._fishers_pp(thetas, ps.xhat, ps.P, u_prev, precs))
                  if self.optimizer in ("fisher", "svn") else None)
            if self.constraint is not None:
                S = self._penalty(S, thetas, z, self._nu)
            force = JtJ = None
            if self.eq is not None:
                pen, JtJ = self._mdmm_terms(thetas, self._lam, aux)
                force = -pen
            tim["filter"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            thetas, info = self._svgd(thetas, S, xbar, Ubar, Fn, force, JtJ)
            if self.constraint is not None:
                z, self._nu = _block(self._admm(thetas, z, self._nu))
            if self.eq is not None:
                self._lam = _block(self._mdmm_lam(thetas, self._lam, aux))
            tim["kernel"] += time.perf_counter() - t0
        if self.constraint is not None:
            thetas = z
        P_aug_new = _block(self._inflate(P_aug_new, thetas))
        self.t += 1
        return (ParticleSet(thetas, xhat, P_aug_new[:, :n_x, :n_x], ps.w, P_aug_new), tim,
                {k: float(v) for k, v in info.items()})


def _initial_P_aug(P, sigma, n_th):
    """blockdiag(P, diag(sigma^2)) per particle (sigma scalar or (n_th,))."""
    s2 = jnp.broadcast_to(jnp.asarray(sigma, float) ** 2, (n_th,))
    return jax.vmap(lambda Pi: jax.scipy.linalg.block_diag(Pi, jnp.diag(s2)))(P)


def systematic_resample(key, w):
    N = w.shape[0]
    positions = (jax.random.uniform(key) + jnp.arange(N)) / N
    idx = jnp.searchsorted(jnp.cumsum(w), positions)
    return jnp.minimum(idx, N - 1)


class RBPF:
    """Rao-Blackwellized particle filter with a random-walk proposal (baseline)."""

    def __init__(self, model, Q, R, *, sigma_rw, ess_frac=0.5, constraint=None, reject=None):
        """constraint: its box part is applied after the proposal (clipping);
        the equalities cannot be imposed this way. sigma_rw: scalar or (n_theta,)."""
        self.model, self.constraint = model, constraint
        self.reject = reject                 # inequalities c(theta) <= 0: violating particles get weight 0
        self.n_all_rejected = 0              # steps at which every particle violated (rejection skipped)
        self._viol = jax.jit(reject.violation) if reject is not None else None
        self._aux = (jax.jit(reject.aux_fn) if reject is not None
                     and getattr(reject, "aux_fn", None) is not None else None)
        self.Q, self.R = jnp.asarray(Q), jnp.asarray(R)
        self.sigma_rw, self.ess_frac = jnp.asarray(sigma_rw, float), float(ess_frac)
        self._ekf = jax.jit(lambda th, x, P, u, y: ekf_particles(model, th, x, P, u, y,
                                                                 self.Q, self.R))
        self._key = jax.random.PRNGKey(0)

    def reset(self, seed=0):
        """Fresh random stream for the proposal and the resampling."""
        self._key = jax.random.PRNGKey(int(seed) + 7919)
        self.n_all_rejected = 0

    def step(self, ps: ParticleSet, u_prev, y, Ubar=None):
        """Filter step with u_{t-1} and y_t (Ubar: unused, same interface as RBSGD)."""
        t0 = time.perf_counter()
        self._key, k_rw, k_rs = jax.random.split(self._key, 3)
        thetas = ps.thetas + self.sigma_rw * jax.random.normal(k_rw, ps.thetas.shape)
        if self.constraint is not None:
            thetas = self.constraint.clip_box(thetas)
        xhat, P, ll = self._ekf(thetas, ps.xhat, ps.P, u_prev, y)
        logw = jnp.log(ps.w) + ll
        logw = jnp.where(jnp.isfinite(logw), logw, -jnp.inf)
        if self.reject is not None:          # rejection: weight 0 unless every particle with a
            aux = None                       # positive weight violates
            if self._aux is not None and Ubar is not None:
                aux = self._aux(ps.w @ xhat, jnp.asarray(Ubar), ps.w @ thetas)
                self.reject.last_aux = aux
            ok = (self._viol(thetas, aux) <= 0.0) & jnp.isfinite(logw)
            if bool(jnp.any(ok)):
                logw = jnp.where(ok, logw, -jnp.inf)
            else:
                self.n_all_rejected += 1
        if not bool(jnp.any(jnp.isfinite(logw))):     # no particle with a finite weight: uniform
            logw = jnp.zeros_like(logw)
        w = jnp.exp(logw - jnp.max(logw))
        w = w / jnp.sum(w)
        ess = float(1.0 / jnp.sum(w ** 2))
        N = w.shape[0]
        resampled = ess < self.ess_frac * N
        if resampled:
            idx = systematic_resample(k_rs, w)
            thetas, xhat, P = thetas[idx], xhat[idx], P[idx]
            w = jnp.full(N, 1.0 / N)
        _block(w)
        tim = {"filter": time.perf_counter() - t0, "kernel": 0.0}
        return ParticleSet(thetas, xhat, P, w), tim, {"ess": ess, "resampled": float(resampled)}

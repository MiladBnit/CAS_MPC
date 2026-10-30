"""Case study: continuous bioreactor with Haldane substrate inhibition and
product-inhibited growth (the benchmark of the RBSVGD paper),
controlled by the dilution rate D.

States x = (X, S, P) (biomass, substrate, product, g/L), input D (1/h):

    mu(S, P)  = eta(t) mu_max S / (K_s + S + S^2/K_i) (1 - P/P_max)_+
    dX/dt     = mu X - k_dp P X - D X
    dS/dt     = -mu X / Y_xs + D (S_f - S)
    dP/dt     = (Y_px mu + beta) X - D P

The product P is measured and tracked; the biomass X is not measured and is
kept in the corridor X_min <= X <= X_max by chance constraints (X_max optional).
The mixing efficiency eta(t) drifts during the run, a sigmoid from eta_start to
eta_end (centre eta_center, width eta_width) plus an optional AR(1) wobble,
floored at 0.05. The true plant integrates the balances with n_sub RK4 substeps
per sample and adds Gaussian process noise Q_true; the measurement noise of P
has variance R_true.

Gray-box model (normalized coordinates, one RK4 step per sample), model.structure:
  "nn":     the mass balances with the specific growth rate learned,
            mu = mu_scale softplus(NN(S, P; theta)), theta the network weights
            (time-varying: the drift of eta is tracked online). The network
            starts untrained (pretrain.enabled false) or is fitted offline on
            full-mixing data (pretrain_mu);
  "eta":    the physics above with theta = (eta,), box eta in [eta_min, 1].
States are ordered X first, so the constraint margins g(x) are affine in x[0].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .constraints import EqualityConstraint
from .model import init_theta, mlp_shapes, n_theta as _n_theta, unflatten
from .normalization import Normalization


def normalization(c) -> Normalization:
    nc = c["norm"]
    return Normalization(np.asarray(nc["mu_x"], float), np.asarray(nc["Lambda_x"], float),
                         np.array([float(nc["mu_u"])]), np.array([float(nc["Lambda_u"])]),
                         np.array([float(nc["mu_x"][2])]), np.array([float(nc["Lambda_x"][2])]))


def mu_true(S, P, eta, c, np_=np):
    """Haldane x product inhibition, times the mixing efficiency."""
    S_ = np_.clip(S, 0.0, 100.0)
    return eta * c["mu_max"] * S_ / (c["Ks"] + S_ + S_ * S_ / c["Ki"]) * np_.maximum(1.0 - P / c["Pmax"], 0.0)


def _rhs(X, S, P, D, eta, c, np_=np):
    return _rhs_mu(X, S, P, D, mu_true(S, P, eta, c, np_), c)


def _rhs_mu(X, S, P, D, mu, c):
    dX = mu * X - c["kdp"] * P * X - D * X
    dS = -mu * X / c["Yxs"] + D * (c["Sf"] - S)
    dP = (c["Ypx"] * mu + c["beta"]) * X - D * P
    return dX, dS, dP


class BioPlant:
    """True bioreactor, state (X, S, P) in g/L."""

    def __init__(self, c: dict):
        self.c = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in c.items()}
        self.t_s, self.n_sub = self.c["t_s"], int(self.c["n_sub"])
        self.q_std = np.sqrt(np.asarray(c["Q_true"], float))
        self.R_true = self.c["R_true"]
        self._eta = None

    def eta(self, t):
        """True mixing efficiency at step t."""
        c = self.c
        t = int(t)
        if self._eta is None or t >= len(self._eta):
            n = max(t + 1, 2000)
            rng = np.random.default_rng(int(c.get("eta_seed", 0)))
            phi = np.exp(-self.t_s / c["eta_wobble_tau"])
            sd = c["eta_wobble_std"] * np.sqrt(1 - phi ** 2)
            w = np.zeros(n)
            for k in range(1, n):
                w[k] = phi * w[k - 1] + sd * rng.standard_normal()
            th = np.arange(n) * self.t_s
            s = 1.0 / (1.0 + np.exp(-(th - c["eta_center"]) / c["eta_width"]))
            self._eta = np.maximum(c["eta_start"] + (c["eta_end"] - c["eta_start"]) * s + w, 0.05)
        return float(self._eta[t])

    def rhs(self, x, D, t):
        return np.array(_rhs(x[0], x[1], x[2], D, self.eta(t), self.c))

    def advance(self, s, u, rng: np.random.Generator | None = None, t: int = 0):
        """One sample (RK4 substeps). Returns the next state and the constrained
        quantity c = -X (violation when c > -X_min)."""
        h = self.t_s / self.n_sub
        x = np.asarray(s, float)
        for _ in range(self.n_sub):
            k1 = self.rhs(x, u, t)
            k2 = self.rhs(x + h / 2 * k1, u, t)
            k3 = self.rhs(x + h / 2 * k2, u, t)
            k4 = self.rhs(x + h * k3, u, t)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if rng is not None:
            x = x + self.q_std * rng.standard_normal(3)
        x = np.maximum(x, 1e-6)
        return x, float(-x[0])

    def measure(self, s, rng: np.random.Generator | None = None):
        y = float(s[2])
        if rng is not None:
            y += np.sqrt(self.R_true) * rng.standard_normal()
        return y

    def steady_state(self, D, eta, n=4000):
        x = np.array([0.5, 4.0, 0.5])
        h = 0.2
        f = lambda z: np.array(_rhs(z[0], z[1], z[2], D, eta, self.c))
        for _ in range(n):
            k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
            x = np.maximum(x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 1e-6)
        return x

    def steady_input(self, P_target, eta, lo=0.005, hi=0.3):
        """Dilution rate whose steady state has product P_target (P falls with D)."""
        for _ in range(50):
            D = 0.5 * (lo + hi)
            if self.steady_state(D, eta, 1500)[2] > P_target:
                lo = D
            else:
                hi = D
        return 0.5 * (lo + hi)


class BioModel:
    """Normalized gray-box model of the bioreactor."""

    n_x = 3

    def __init__(self, c: dict, norm: Normalization, structure="eta", integrator="rk4"):
        self.c = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in c.items()}
        self.norm = norm
        if structure not in ("eta", "nn"):
            raise ValueError(structure)
        self.structure = structure
        self.integrator = integrator
        nm = self.c.get("nn", {})
        self.shapes = mlp_shapes(int(nm.get("n_hidden", 8)), int(nm.get("n_layers", 1)), 1, n_in=2)
        self.mu_scale = float(nm.get("mu_scale", 0.2))
        self.n_theta = 1 if structure == "eta" else _n_theta(self.shapes)
        self._mu = [float(v) for v in norm.mu_x]
        self._La = [float(v) for v in norm.Lambda_x]
        self._mu_u, self._La_u = float(norm.mu_u[0]), float(norm.Lambda_u[0])
        self.X_min_norm = (self.c["X_min"] - self._mu[0]) / self._La[0]
        xm = self.c.get("X_max")                    # optional upper bound: the corridor X_min <= X <= X_max
        self.X_max_norm = None if xm is None else (float(xm) - self._mu[0]) / self._La[0]
        self.n_c = 1 if xm is None else 2

    def eta_of(self, theta):
        return jnp.clip(theta[0], 1e-3, 2.0)

    def mu(self, x, theta):
        """Specific growth rate (1/h) at the normalized state x."""
        S = self._mu[1] + self._La[1] * x[1]
        P = jnp.maximum(self._mu[2] + self._La[2] * x[2], 0.0)
        if self.structure == "eta":
            return mu_true(S, P, self.eta_of(theta), self.c, np_=jnp)
        z = jnp.stack([x[1], x[2]])
        layers = unflatten(theta, self.shapes)
        for W, b in layers[:-1]:
            z = jnp.tanh(W @ z + b)
        W, b = layers[-1]
        return self.mu_scale * jax.nn.softplus((W @ z + b)[0])

    def _incr(self, x, u, theta):
        X, S, P = (self._mu[i] + self._La[i] * x[i] for i in range(3))
        D = self._mu_u + self._La_u * u
        dX, dS, dP = _rhs_mu(jnp.maximum(X, 0.0), S, jnp.maximum(P, 0.0), D, self.mu(x, theta),
                             self.c)
        ts = self.c["t_s"]
        return jnp.stack([ts * dX / self._La[0], ts * dS / self._La[1], ts * dP / self._La[2]])

    def f(self, x, u, theta):
        v = lambda z: self._incr(z, u, theta)
        if self.integrator == "euler":
            return x + v(x)
        k1 = v(x)
        k2 = v(x + 0.5 * k1)
        k3 = v(x + 0.5 * k2)
        k4 = v(x + k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def h(self, x):
        """Measured (and tracked) output: normalized P (y shares the normalization of P)."""
        return x[2]

    def g(self, x):
        """X_min - X <= 0 (normalized); with X_max also X - X_max <= 0 (vector of 2)."""
        lo = self.X_min_norm - x[0]
        if self.X_max_norm is None:
            return lo
        return jnp.stack([lo, x[0] - self.X_max_norm])

    def rollout(self, x0, U, theta):
        def step(x, u):
            xn = self.f(x, u, theta)
            return xn, xn
        _, xs = jax.lax.scan(step, x0, U)
        return jnp.concatenate([x0[None], xs])

    def initial_theta(self, key=None):
        if self.structure == "nn":
            if not self.c["pretrain"].get("enabled", True):     # untrained: constant mu = mu_scale ln 2
                return init_theta(jax.random.PRNGKey(int(self.c["pretrain"]["seed"])), self.shapes, 0.0)
            return pretrain_mu(self, self.c)
        return jnp.array([float(self.c["eta_prior"])])


def pretraining_data(c):
    """Offline data at full mixing (eta = 1): the plant under random steps of the
    dilution rate (c["pretrain"]: n_steps samples, steps held hold_min..hold_max
    samples, D in [D_lo, D_hi]); X, S, P all assayed (measurement noise
    c["pretrain"]["noise_std"] relative). Returns states (n+1, 3) and inputs (n,)."""
    pc = c["pretrain"]
    rng = np.random.default_rng(int(pc["seed"]))
    plant = BioPlant(dict(c, eta_start=1.0, eta_end=1.0, eta_wobble_std=0.0))
    n = int(pc["n_steps"])
    D = np.zeros(n)
    k = 0
    while k < n:
        h = int(rng.integers(pc["hold_min"], pc["hold_max"] + 1))
        D[k:k + h] = rng.uniform(pc["D_lo"], pc["D_hi"])
        k += h
    x = plant.steady_state(float(D[0]), 1.0)
    X = [x]
    for k in range(n):
        x, _ = plant.advance(x, float(D[k]), rng)
        X.append(x)
    X = np.array(X)
    X = X * (1 + float(pc["noise_std"]) * rng.standard_normal(X.shape))
    return np.maximum(X, 1e-6), D


def pretrain_mu(model: BioModel, c):
    """Fit the growth-rate network on the pre-training data (normalized states):
    simulation error over windows of pretrain.window samples, Adam then L-BFGS."""
    pc = c["pretrain"]
    Xp, D = pretraining_data(c)
    Xn = jnp.asarray(model.norm.x_norm(Xp))
    Un = jnp.asarray(model.norm.u_norm(D))
    th0 = init_theta(jax.random.PRNGKey(int(pc["seed"])), model.shapes, 0.1)

    # multi-step simulation error over windows of K samples from the measured
    # window start states (the subsequence approach of the normalization paper)
    K = int(pc.get("window", 20))
    starts = jnp.arange(0, len(Un) - K, K // 2)
    Uw = jnp.stack([Un[i:i + K] for i in range(0, len(Un) - K, K // 2)])
    Xw = jnp.stack([Xn[i + 1:i + K + 1] for i in range(0, len(Un) - K, K // 2)])
    X0 = Xn[starts]

    law = inhibition_laws(model, c)
    w_law = float(pc.get("law_penalty", 0.0))

    def loss(th):
        pred = jax.vmap(lambda x0, us: model.rollout(x0, us, th)[1:])(X0, Uw)
        out = jnp.mean((pred - Xw) ** 2) + float(pc.get("weight_decay", 1e-6)) * jnp.sum(th ** 2)
        return out + w_law * jnp.sum(jnp.maximum(law.c(th), 0.0) ** 2)      # the grid

    opt = optax.adam(float(pc.get("lr", 1e-2)))
    st = opt.init(th0)

    @jax.jit
    def step(th, st):
        v, g = jax.value_and_grad(loss)(th)
        up, st = opt.update(g, st, th)
        return optax.apply_updates(th, up), st, v

    th = th0
    for _ in range(int(pc.get("adam_iters", 2000))):
        th, st, _ = step(th, st)
    lb = optax.lbfgs()
    ls = lb.init(th)
    vg = optax.value_and_grad_from_state(loss)

    @jax.jit
    def lstep(th, ls):
        v, g = vg(th, state=ls)
        up, ls = lb.update(g, ls, th, value=v, grad=g, value_fn=loss)
        return optax.apply_updates(th, up), ls

    for _ in range(int(pc.get("lbfgs_iters", 300))):
        th, ls = lstep(th, ls)
    return th


def inhibition_laws(model: BioModel, c, rho=1.0, eta=1.0):
    """Substrate inhibition above S* = sqrt(K_s K_i) and product inhibition of
    the learned growth rate, d mu/dS <= 0 and d mu/dP <= 0 (exact derivatives
    of the network), at the collocation grid c["laws"]: S in S_range, P in
    P_range (n_S x n_P points, S_range above S*), plus two moving points
    computed by the filter each step: its state estimate and the end of the
    prediction under the shifted plan. Inequalities, MDMM for the Stein
    filters, rejection for RBPF. True at any mixing level (eta only scales mu)."""
    lc = c["laws"]
    Sg = np.linspace(*lc["S_range"], int(lc["n_S"]))
    Pg = np.linspace(*lc["P_range"], int(lc["n_P"]))
    G = np.array([[model._mu[0], S, P] for S in Sg for P in Pg])
    Xn = jnp.asarray(model.norm.x_norm(G))
    sS, sP = 1.0 / model._La[1], 1.0 / model._La[2]

    def slopes(theta, X):
        d = jax.vmap(jax.grad(lambda x: model.mu(x, theta)))(X)
        return jnp.concatenate([d[:, 1] * sS, d[:, 2] * sP])

    def cfun(theta):                                  # the grid only
        return slopes(theta, Xn)

    def c_aux(theta, aux):                            # the grid and the moving points
        return jnp.concatenate([slopes(theta, Xn), slopes(theta, aux)])

    def aux_fn(xbar, Ubar, theta_mean):
        """Moving collocation points: the state estimate and the end of the
        prediction under the shifted plan (towards washout)."""
        return jnp.stack([xbar, model.rollout(xbar, Ubar, theta_mean)[-1]])
    return EqualityConstraint(cfun, 2 * (len(G) + 2), rho=rho, eta=eta, inequality=True,
                              c_aux=c_aux, aux_fn=aux_fn)

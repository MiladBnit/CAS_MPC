"""Filters and controllers of the bioreactor case study, with the interface used
by closed_loop.run_closed_loop, and the metrics of a closed-loop run (bio_summary).

Methods: "param" and "ca" (RBSGD with the parameter kernel k_theta or the
control-aware kernel k_omega, eq:kernels) with the scenario SMPC (eq:smpc);
"rbpf" (Rao-Blackwellized particle filter) with the scenario SMPC; "ce" (CE-MPC
on the particle mean of the "ca" filter); "ca0" (ablation: the "ca" filter with
omega = 0, i.e. k_Phi alone). With filter.constrained, the inhibition
laws on the learned growth rate (bioreactor.inhibition_laws) are imposed by MDMM
in the Stein filters and by rejection in RBPF.

Constraint set: X_min <= X (washout) and, if bio.X_max is set, X <= X_max (one
chance constraint per side, model.g).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .bioreactor import BioModel, BioPlant, inhibition_laws, mu_true, normalization
from .constraints import Constraint
from .metrics import crps_mixture
from .filters import RBPF, RBSGD, ParticleSet
from .mpc import ScenarioMPC


class BioSetup:
    def __init__(self, cfg, methods):
        self.cfg, self.methods = cfg, tuple(methods)
        c, fc, rc, mc = cfg["bio"], cfg["filter"], cfg["rbpf"], cfg["mpc"]
        self.norm = normalization(c)
        self.model = BioModel(c, self.norm, c["model"]["structure"], c["model"]["integrator"])
        self.nn = self.model.structure == "nn"
        self.keep_thetas = True                       # for the growth-rate tracking metrics
        self.box = (None if self.nn else
                    Constraint(self.model.n_theta, box_idx=[0], lo=[float(c["eta_min"])], hi=[1.0]))
        # recorded: the growth rate at a reference state (the setpoint region), model and truth
        self._ref = jnp.asarray(self.norm.x_norm(np.asarray(c.get("mu_ref_state", [0.35, 4.3, 0.45]), float)))
        self._mu_ref = jax.jit(jax.vmap(lambda th: self.model.mu(self._ref, th)))
        self.theta_0 = self.model.initial_theta()
        self.laws = (inhibition_laws(self.model, c, fc["mdmm_rho"], fc["mdmm_eta"])
                     if self.nn else None)            # physical laws on the learned growth rate
        self._law_viol = jax.jit(self.laws.violation) if self.laws is not None else None   # (thetas, aux)
        self.sigma = jnp.full(self.model.n_theta, float(fc["sigma_init_nn" if self.nn else "sigma_init"]))
        La = self.norm.Lambda_x
        # noise of the filters and the SMPC: Q_model, R_model (physical, per sample) if given,
        # else the plant's Q_true, R_true times Q_scale, R_scale
        Qm = (np.asarray(fc["Q_model"], float) if fc.get("Q_model") is not None
              else np.asarray(c["Q_true"], float) * fc["Q_scale"])
        Rm = (float(fc["R_model"]) if fc.get("R_model") is not None
              else c["R_true"] * fc.get("R_scale", 1.0))
        self.Q = jnp.diag(jnp.asarray(Qm / La ** 2))
        self.R = jnp.array([[Rm / La[2] ** 2]])
        self.limit = -float(c["X_min"])                 # violation: -X > -X_min
        self.plant = BioPlant(c)
        P0 = float(cfg["closed_loop"]["reference"][0][0])
        self.u0 = self.plant.steady_input(P0, self.plant.eta(0))
        self.x_true0 = self.plant.steady_state(self.u0, self.plant.eta(0))
        x0_hat_p = self.plant.steady_state(self.u0, 1.0 if self.nn else float(c["eta_prior"]))
        self.x0_hat = jnp.asarray(self.norm.x_norm(x0_hat_p))
        if self.nn:
            # the plant starts at steady state at the known input u0 and product P0 (the first reference):
            # the biomass balance gives its growth rate, mu0 = u0 + k_dp P0, and the product and substrate
            # balances its biomass and substrate, without knowing eta. The state estimate starts there and
            # the untrained network (zero last layer) at the constant mu0, so the model is stationary at t = 0.
            pc = self.plant.c
            mu0 = self.u0 + pc["kdp"] * P0
            X0 = self.u0 * P0 / (pc["Ypx"] * mu0 + pc["beta"])
            S0 = pc["Sf"] - mu0 * X0 / (pc["Yxs"] * self.u0)
            self.x0_hat = jnp.asarray(self.norm.x_norm(np.array([X0, S0, P0])))
            if not c["pretrain"].get("enabled", True):
                v = mu0 / self.model.mu_scale
                self.theta_0 = self.theta_0.at[-1].set(float(np.log(np.expm1(v))))   # softplus^-1
        self.P0 = jnp.diag(jnp.asarray(c["P0"], float))

        u_lo, u_hi = (float(self.norm.u_norm(c[k])) for k in ("u_min", "u_max"))
        self.filters, self.mpcs = {}, {}
        for m in self.methods:
            on = fc.get("constrained", True)
            if m == "rbpf":
                self.filters[m] = RBPF(self.model, self.Q, self.R,
                                       sigma_rw=rc["sigma_rw_nn" if self.nn else "sigma_rw"],
                                       ess_frac=rc["ess_frac"], constraint=self.box if on else None,
                                       reject=self.laws if on and self.nn else None)
            else:
                self.filters[m] = RBSGD(                      # ce, ca0: the ca filter; ca0 with omega = 0
                    self.model, "param" if m == "param" else "ca", self.Q, self.R, M=fc["M"],
                    epsilon=fc["epsilon"], omega=0.0 if m == "ca0" else fc["omega"],
                    lambda_Sigma=fc["lambda_Sigma"],
                    theta_0=self.theta_0, sigma_init=self.sigma,
                    L_Phi=int(mc["L"] if fc.get("L_Phi") is None else fc["L_Phi"]),
                    W_y=mc["W_y"], h_floor=fc["h_floor"], optimizer=fc.get("optimizer", "sgd"),
                    alpha=fc.get("alpha", 1.0),
                    scale_h=(fc["scale_h_param"] if m == "param" and fc.get("scale_h_param") is not None
                             else fc.get("scale_h", 1.0)),           # param may have its own bandwidth scale
                    scale_h_phi=fc.get("scale_h_phi"),
                    score_mode=fc.get("score", "marginal"),
                    score_window=int(fc.get("score_window", 1)),
                    Q_theta=fc.get("Q_theta_nn" if self.nn else "Q_theta", 0.0),
                    lr_ekf=fc.get("lr_ekf", 1.0),
                    prior=fc.get("prior", "particles"),
                    fisher_beta1=fc.get("fisher_beta1", 0.9), fisher_beta2=fc.get("fisher_beta2", 0.999),
                    fisher_damping=fc.get("fisher_damping", 1e-3),
                    fisher_max_step=fc.get("fisher_max_step"),
                    constraint=self.box if on and not self.nn else None,
                    eq_constraint=self.laws if on and self.nn else None)
            self.mpcs[m] = ScenarioMPC(
                self.model, N=1 if m == "ce" else self.n_particles(m), L=mc["L"], W_y=mc["W_y"],
                W_u=mc["W_u"], delta=mc["delta"], s0=mc["s0"], Q=np.asarray(self.Q),
                u_min=u_lo, u_max=u_hi, rho_1=mc["rho_1"], rho_2=mc["rho_2"],
                mode="ce" if m == "ce" else "smpc", ipopt=mc.get("ipopt"))

    def n_particles(self, method):
        return int(self.cfg["rbpf"]["N"] if method == "rbpf" else self.cfg["filter"]["N"])

    def make_plant(self):
        return BioPlant(self.cfg["bio"])

    def initial_particles(self, key, N, project=True):
        th = self.theta_0 + self.sigma * jax.random.normal(key, (N, self.model.n_theta))
        if self.box is not None:
            th = self.box.project(th)
        return ParticleSet(th, jnp.tile(self.x0_hat, (N, 1)), jnp.tile(self.P0, (N, 1, 1)),
                           jnp.full(N, 1.0 / N))

    def constrained(self, x):
        return float(-x[0])

    def near_limit(self, x):
        """Steps whose true biomass is within 0.05 g/L of X_min (or of X_max)."""
        X = np.asarray(x)[:, 0]
        near = X < -self.limit + 0.05
        xm = self.cfg["bio"].get("X_max")
        return near if xm is None else near | (X > float(xm) - 0.05)

    def violated(self, x):
        """X < X_min, or X > X_max when the corridor has an upper bound."""
        X = np.asarray(x)[:, 0]
        v = X < -self.limit
        xm = self.cfg["bio"].get("X_max")
        return v if xm is None else v | (X > float(xm))

    def record_params(self, thetas):
        """Per particle: the growth rate at the reference state relative to the
        full-mixing one (= eta for structure eta), twice, and the box violation (N, 3)."""
        if self.nn:                           # growth ratio, twice, largest law violation
            r = np.asarray(self._mu_ref(thetas)) / self._mu_ref_fresh
            return np.column_stack([r, r, np.asarray(self._law_viol(thetas, self.laws.last_aux))])
        e = np.asarray(thetas)[:, 0]
        res = np.maximum(0.0, np.maximum(e - 1.0, self.box.lo[0] - e))
        return np.column_stack([e, e, res])

    @property
    def _mu_ref_fresh(self):
        c = self.cfg["bio"]
        S, P = c.get("mu_ref_state", [0.35, 4.3, 0.45])[1:]
        return float(mu_true(S, P, 1.0, BioPlant(c).c))


def growth_tracking(res, setup):
    """Learned growth rate at the true state along the trajectory: particles
    (T, N) and their (weighted) mean (T,), and the true growth rate (T,)."""
    c = setup.plant.c
    X = res["x"]
    true = np.array([float(mu_true(x[1], x[2], setup.plant.eta(k), c)) for k, x in enumerate(X)])
    Xn = jnp.asarray(setup.norm.x_norm(X))
    TH = jnp.asarray(res["thetas"])
    mu = np.asarray(jax.vmap(lambda th_t, x: jax.vmap(lambda th: setup.model.mu(x, th))(th_t))(TH, Xn))
    w = res["w"] / res["w"].sum(axis=1, keepdims=True)
    return mu, np.einsum("tn,tn->t", w, mu), true


def bio_summary(res, setup):
    """The metrics of one closed-loop run:

    washouts: the number of steps with X < X_min (with X_max, also X > X_max; then also
      per side, washouts low and high);
    tracking RMSE: root mean of (P - r)^2 over the steps without a washout, g/L (a
      controller gets no credit for tracking while it violates; a mean, as the
      methods keep different numbers of steps);
    CRPS X, CRPS P: of the forecasts the controller uses (mixture of the scenarios'
      Gaussians), mean over the steps of the horizon and over time (g/L);
    CRPS mu (structure nn): of the particles' growth rates at the true state (no
      per-particle variance), mean over time (1/h);
    computation ms: filter and MPC per step (t = 0, the JIT compilation, excluded).
    CRPS: lower is better; it rewards forecasts that are both accurate and sharp
    (metrics.crps_mixture)."""
    x, w = res["x"], res["w"] / res["w"].sum(axis=1, keepdims=True)
    v = np.asarray(res["violation"], bool)
    e2 = (res.get("z", res["y"]) - res["r"]) ** 2
    s = {"washouts": int(v.sum())}
    xm = setup.cfg["bio"].get("X_max")
    if xm is not None:                                # the corridor: which side
        s["washouts low"] = int(np.sum(x[:, 0] < -setup.limit))
        s["washouts high"] = int(np.sum(x[:, 0] > float(xm)))
    s["tracking RMSE"] = float(np.sqrt(e2[~v].mean())) if (~v).any() else float("nan")
    if "pred_std" in res:                             # the controller's forecasts over the horizon
        L = res["pred"].shape[2]
        ws = res["w"] if res["pred"].shape[1] == res["w"].shape[1] else np.ones(res["pred"].shape[:2])
        ws = ws / ws.sum(axis=1, keepdims=True)
        for k, nm in ((0, "X"), (2, "P")):
            s[f"CRPS {nm}"] = float(np.mean([np.mean(crps_mixture(
                res["pred"][:-l, :, l - 1, k], res["pred_std"][:-l, :, l - 1, k], ws[:-l], x[l:, k]))
                for l in range(1, L + 1)]))
    if setup.nn and "thetas" in res:                  # the parameter posterior: growth rates at the true state
        mu_p, _, mu_t = growth_tracking(res, setup)
        s["CRPS mu"] = float(np.mean(crps_mixture(mu_p, np.zeros_like(mu_p), w, mu_t)))
    s["computation ms"] = 1e3 * float(res["t_filter"][1:].mean() + res["t_mpc"][1:].mean())
    return s

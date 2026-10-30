"""Closed loop: true plant + particle filter + scenario SMPC (paper, Algorithm alg:ca).

At every time t:
  1. measure y_t on the true plant (physical units), normalize;
  2. filter step with (u_{t-1}, y_t); the Stein filters use the shifted plan
     Ubar_t (eq:shifted-plan) in the feature map. At t = 0 the plant is at the
     steady state of u_{-1} = u0, so the same step with u0 is exact;
  3. solve the SMPC (or CE-MPC) warm-started at Ubar_t, with the reference
     preview r_{t+1..t+L}; if IPOPT fails, the shifted previous plan Ubar_t is
     applied instead;
  4. apply u*_{t|t}, mapped to physical units, and advance the plant.

Times are recorded per step; the first step includes the JIT compilation.

The case study is the bioreactor (bio_setup.BioSetup): it provides the model,
filters, controllers, make_plant(), x_true0, u0, initial_particles(key, N),
constrained(x), limit, violated(x), near_limit(x), ref_norm and tracked.

Methods:
  "ce"          CE-MPC on the particle mean of the proposed filter ("ca")
  "param", "ca" RBSGD with that kernel + scenario SMPC
  "ca0"         ablation: "ca" with omega = 0 (k_Phi alone) + scenario SMPC
  "rbpf"        RBPF (baseline) + scenario SMPC with the particle weights
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .features import shifted_plan

METHODS = ("ce", "param", "ca", "ca0", "rbpf")


def input_profile(segments):
    """[[value, n_steps], ...] -> piecewise-constant sequence."""
    return np.concatenate([np.full(int(n), float(v)) for v, n in segments])


def make_setup(cfg, methods):
    """Setup of the case study (config case: bioreactor)."""
    if cfg.get("case") != "bioreactor":
        raise ValueError(f"unknown case {cfg.get('case')!r} (expected 'bioreactor')")
    from .bio_setup import BioSetup
    return BioSetup(cfg, methods)


def _controller_view(method, ps):
    """Scenarios passed to the controller: CE uses the particle mean, P = 0."""
    if method == "ce":
        w = np.asarray(ps.w)
        return (np.asarray(w @ np.asarray(ps.thetas))[None],
                np.asarray(w @ np.asarray(ps.xhat))[None], np.zeros((1,) + ps.P.shape[1:]),
                np.ones(1))
    return np.asarray(ps.thetas), np.asarray(ps.xhat), np.asarray(ps.P), np.asarray(ps.w)


def run_closed_loop(setup, method: str, seed: int, progress=None):
    """One run. ``progress``: optional callback progress(t, T).

    Returns physical-unit trajectories: the true states x (T, 3), c_free (T,) the
    constrained quantity, violation (T,) (setup.violated), near (T,) the steps
    near a bound, y (T,), u (T,), r (T,), z (T,) the tracked quantity, the
    particles' state estimates xhat_p (T, N, 3) and standard deviations xstd_p
    after the filter update, the particle weights w (T, N) (uniform except RBPF),
    the particles' parameters thetas (T, N, n_theta) and at the end theta_final,
    w_final, the controller's scenario predictions under the chosen plan pred
    (T, N_s, L, 3) (pred[t, i, l-1] predicts the state at t + l; N_s = 1 for
    CE-MPC) and their standard deviations pred_std, the predicted probability of
    a violation at t + 1 risk (T,) and per constraint risk_c (T, n_c), recorded
    parameters params, and per step: time (s) of the filter and of the MPC solve,
    IPOPT iterations, whether a slack was active, solver failures."""
    cfg, norm = setup.cfg, setup.norm
    cl = cfg["closed_loop"]
    L = cfg["mpc"]["L"]
    r_p = input_profile(cl["reference"])
    T = len(r_p)
    r_n = np.asarray(setup.ref_norm(r_p) if hasattr(setup, "ref_norm") else norm.y_norm(r_p))
    r_pad = np.concatenate([r_n, np.full(L + 1, r_n[-1])])

    plant = setup.make_plant()
    rng = np.random.default_rng(cl["seed_offset"] + seed)
    k0 = jax.random.PRNGKey(cl["seed_offset"] + seed)
    try:
        ps = setup.initial_particles(k0, setup.n_particles(method), project=not method.endswith("_uc"))
    except TypeError:
        ps = setup.initial_particles(k0, setup.n_particles(method))
    flt, mpc = setup.filters[method], setup.mpcs[method]
    flt.reset(seed=cl["seed_offset"] + seed)
    record_params = getattr(setup, "record_params", None)

    x = np.array(setup.x_true0, float)
    c_free = setup.constrained(x) if hasattr(setup, "constrained") else float(x[0])
    u_prev = float(norm.u_norm(setup.u0))
    U_prev = np.full(L, u_prev)
    rec = {k: [] for k in ("x", "c_free", "y", "u", "xhat_p", "xstd_p", "w", "pred", "pred_std", "risk", "risk_c",
                           "t_filter", "t_mpc", "iters", "slack", "fail", "params")}

    for t in range(T):
        y_p = plant.measure(x, rng)
        y = float(norm.y_norm(y_p))
        Ubar = np.asarray(shifted_plan(jnp.asarray(U_prev))) if t > 0 else U_prev
        ps, tim, _ = flt.step(ps, u_prev, y, jnp.asarray(Ubar))
        t_f = tim["filter"] + tim["kernel"]
        view = _controller_view(method, ps)
        sol = mpc.solve(*view, r_pad[t + 1:t + L + 1], u_prev, Ubar)
        U = sol["U"] if sol["success"] and np.all(np.isfinite(sol["U"])) else Ubar
        mu, Sig = mpc.predict(U, *view[:3])                        # the controller's prediction
        npx = len(norm.mu_x)                                       # physical states
        pred = norm.mu_x + norm.Lambda_x * mu[..., :npx]           # (N_s, L, npx), physical
        pred_std = norm.Lambda_x * np.sqrt(np.maximum(               # its standard deviations
            np.diagonal(np.asarray(Sig), axis1=-2, axis2=-1)[..., :npx], 0))
        risk_c = 1.0 - np.atleast_1d(mpc.prob(U, *view)[0])        # P(violation at t + 1), per constraint
        risk = float(np.sum(risk_c))                               # (a corridor: disjoint events, exact)
        u_prev, U_prev = float(U[0]), U

        xh = norm.x_phys(np.asarray(ps.xhat)[:, :npx])                          # (N, npx)
        xs = norm.Lambda_x * np.sqrt(np.maximum(
            np.diagonal(np.asarray(ps.P), axis1=1, axis2=2)[:, :npx], 0))
        if record_params is not None:
            rec["params"].append(record_params(ps.thetas))
        if getattr(setup, "keep_thetas", False):
            rec.setdefault("thetas", []).append(np.asarray(ps.thetas))
        for k, v in (("x", x), ("c_free", c_free), ("y", y_p), ("u", norm.u_phys(u_prev)),
                     ("xhat_p", xh), ("xstd_p", xs), ("w", np.asarray(ps.w)), ("pred", pred), ("pred_std", pred_std),
                     ("risk", risk), ("risk_c", risk_c),
                     ("t_filter", t_f), ("t_mpc", sol["time"]), ("iters", sol["iters"]),
                     ("slack", sol["slack_active"]),
                     ("fail", not sol["success"])):
            rec[k].append(v)
        x, c_free = plant.advance(x, float(norm.u_phys(u_prev)), rng, t)
        if progress is not None:
            progress(t, T)

    res = {k: np.array(v) for k, v in rec.items()}
    res["r"] = r_p
    res["theta_final"], res["w_final"] = np.asarray(ps.thetas), np.asarray(ps.w)
    res["z"] = (np.array([setup.tracked(x) for x in res["x"]]) if hasattr(setup, "tracked")
                else res["y"])                               # the tracked quantity (true)
    res["violation"] = (setup.violated(res["x"]) if hasattr(setup, "violated")
                        else res["c_free"] > setup.limit)
    res["near"] = setup.near_limit(res["x"])
    if not len(res["params"]):
        del res["params"]
    return res


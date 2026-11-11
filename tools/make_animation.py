"""Animated closed loop of the bioreactor: the closed-loop figure of the paper, frame by frame (README animation).

One row per method and the columns of the figure: biomass X with the bounds, product P with the reference, and
the learned growth rate mu at the true state. Every frame shows a time window that follows the current time:
the truth (and the measurements of P) up to it; for X and P the forecast the controller uses over its horizon
(the scenario means and the 95% band of their Gaussian mixture, starting from the filtered estimate); for mu the
particles. Between two samples the window slides and the lines are drawn on in `sub` sub-frames; the forecasts,
the measurements and the metrics change at the samples, as in the closed loop.

Each row also shows running metrics, computed from the forecasts whose outcome is already known (the definitions
of bio_setup.bio_summary, so the last frame shows the run's values): the violations, the CRPS of the forecasts of
X and P, the tracking RMSE, and the CRPS of the particles' growth rates.

Line widths and marker sizes are those of the paper figure scaled by the width ratio of the two figures. The output
format follows the extension: .webp (lossy, small) or .gif (one palette shared by all frames). The runs are those
of notebooks/run_bioreactor.ipynb (loaded from results/bioreactor_runs/, or run if missing).

    python tools/make_animation.py                                 # -> assets/closed_loop.webp
    python tools/make_animation.py --out assets/closed_loop.gif --sub 1 --fps 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from PIL import Image  # noqa: E402

import carbsgd.plots as pp  # noqa: E402
from carbsgd.bio_mc import run_cached  # noqa: E402
from carbsgd.bio_setup import bio_summary, growth_tracking  # noqa: E402
from carbsgd.config import CONFIG_DIR, ROOT, load_config  # noqa: E402
from carbsgd.metrics import crps_mixture, mixture_quantiles  # noqa: E402

METHODS = ("ca", "ca0", "param", "rbpf")
ROWS = {"ca": "CAS", "ca0": "CAS\n" + r"$\omega = 0$", "param": "RBSVGD", "rbpf": "RBPF", "ce": "CE-MPC"}
RC_LINES = ("axes.linewidth", "grid.linewidth", "lines.linewidth", "xtick.major.width", "ytick.major.width",
            "xtick.major.size", "ytick.major.size")


def forecasts(r, k, level=0.95):
    """Per time t, for state k: the scenario means over the horizon, starting at the filtered estimate
    (T, N, L + 1), and the band of their mixture (2, T, L + 1)."""
    w = r["w"] / r["w"].sum(axis=1, keepdims=True)
    mean = np.concatenate([r["xhat_p"][:, :, None, k], r["pred"][..., k]], axis=2)
    std = np.concatenate([r["xstd_p"][:, :, None, k], r["pred_std"][..., k]], axis=2)
    T, N, L1 = mean.shape
    flat = lambda a: a.transpose(0, 2, 1).reshape(T * L1, N)
    q = mixture_quantiles(flat(mean), flat(std), np.repeat(w, L1, axis=0), (0.5 * (1 - level), 0.5 * (1 + level)))
    return mean, q.reshape(2, T, L1)


def running_metrics(r, mu):
    """Per sample k, the metrics of bio_summary over what is known at k: CRPS of the forecasts of X and P (for each
    lead l the mean over the forecasts made at s <= k - l, then the mean over l), tracking RMSE over the steps
    without a violation, CRPS of the particles' growth rates, violations. Arrays of length T (NaN until defined)."""
    T, L = r["pred"].shape[0], r["pred"].shape[2]
    w = r["w"] / r["w"].sum(axis=1, keepdims=True)
    out = {}
    for name, s in (("X", 0), ("P", 2)):
        per_lead = np.full((L, T), np.nan)                      # running mean of lead l at time k
        for l in range(1, L + 1):
            c = crps_mixture(r["pred"][:-l, :, l - 1, s], r["pred_std"][:-l, :, l - 1, s], w[:-l], r["x"][l:, s])
            per_lead[l - 1, l:] = np.cumsum(c) / np.arange(1, len(c) + 1)
        with np.errstate(all="ignore"):
            out[name] = np.nanmean(np.where(np.isnan(per_lead), np.nan, per_lead), axis=0)
    mu_p, _, mu_t = mu
    c_mu = crps_mixture(mu_p, np.zeros_like(mu_p), w, mu_t)
    out["mu"] = np.cumsum(c_mu) / np.arange(1, T + 1)
    v = np.asarray(r["violation"], bool)
    e2 = np.where(v, 0.0, (r.get("z", r["y"]) - r["r"]) ** 2)
    n = np.cumsum(~v)
    with np.errstate(all="ignore"):
        out["rmse"] = np.where(n > 0, np.sqrt(np.cumsum(e2) / np.maximum(n, 1)), np.nan)
    out["violations"] = np.cumsum(v)
    return out


def _scaled(style, s):
    return {k: (v * s if k in ("lw", "ms", "mew") else v) for k, v in style.items()}


def _fmt(v, scale):
    return "–" if not np.isfinite(v) else f"{scale * v:.2f}"


def animate(runs, setup, methods, out, fps=16, sub=2, dpi=110, window=60.0, figsize=(10.0, 6.8), colors=48,
            quality=80):
    c = setup.cfg["bio"]
    t_s, L = float(c["t_s"]), int(setup.cfg["mpc"]["L"])
    T = len(runs[methods[0]]["x"])
    t = np.arange(T) * t_s
    s = figsize[0] / pp.TEXT_WIDTH                                 # line widths: the figure's, scaled with the size
    st = pp.FILTER_STYLE
    true_st, meas_st = _scaled(pp.STYLE["true"], s), _scaled(pp.MEASURED, s)
    bound_st, ref_st = _scaled(pp.BOUND, s), _scaled(pp.REFERENCE, s)
    part_st, viol_st = _scaled(pp.PARTICLES, s), _scaled(pp.VIOLATION_MARKER, s)
    band_kw = dict(color=st["fill"], alpha=pp.BAND_ALPHA * st["fill_alpha"], lw=0, zorder=st["fill_zorder"])
    fc = {m: {k: forecasts(runs[m], k) for k in (0, 2)} for m in methods}
    mu = {m: growth_tracking(runs[m], setup) for m in methods}                  # particles, mean, truth
    met = {m: running_metrics(runs[m], mu[m]) for m in methods}
    rc = {**pp.PAPER_RC, "figure.dpi": dpi, "savefig.dpi": dpi, "savefig.bbox": "standard"}
    rc.update({k: rc.get(k, plt.rcParams[k]) * s for k in RC_LINES})

    with plt.rc_context(rc):
        fig, ax = plt.subplots(len(methods), 3, sharex=True, sharey="col", figsize=figsize)
        ax = np.atleast_2d(ax)
        txt = dict(transform=None, ha="left", va="top", fontsize=8, color="#333333", zorder=14)
        art = {}
        for row, m in enumerate(methods):
            r, A = runs[m], {}
            a = ax[row, 0]                                           # biomass
            a.axhline(float(c["X_min"]), **bound_st)
            if c.get("X_max") is not None:
                a.axhline(float(c["X_max"]), **bound_st)
            A["viol"], = a.plot([], [], transform=a.get_xaxis_transform(), **viol_st)
            a.set_ylabel(ROWS.get(m, m), fontsize=9, linespacing=1.1)
            ax[row, 1].step(t, r["r"], where="post", **ref_st)      # product
            A["meas"], = ax[row, 1].plot([], [], **meas_st)
            N = r["pred"].shape[1]
            for col, k in ((0, 0), (1, 2)):
                A[f"band{col}"] = ax[row, col].fill_between([], [], [], **band_kw)
                A[f"fan{col}"] = ax[row, col].plot(np.zeros((2, N)), np.full((2, N), np.nan), color=st["color"],
                                                   **part_st, zorder=st["fill_zorder"] + 1)
                A[f"true{col}"], = ax[row, col].plot([], [], **true_st)
            A["mu"] = ax[row, 2].plot(np.zeros((2, N)), np.full((2, N), np.nan), color=st["color"],
                                      **part_st, zorder=st["fill_zorder"] + 1)            # growth rate
            A["true2"], = ax[row, 2].plot([], [], **true_st)
            A["now"] = [ax[row, j].axvline(0, color="0.55", lw=0.6 * s, ls=":", zorder=2) for j in range(3)]
            A["text"] = [ax[row, j].text(0.015, 0.955, "", **{**txt, "transform": ax[row, j].transAxes})
                         for j in range(3)]
            art[m] = A
        ax[0, 0].set_title(r"biomass $X$ (g/L)")
        ax[0, 1].set_title(r"product $P$ (g/L)")
        ax[0, 2].set_title(r"growth rate $\mu$ (1/h)")
        # fixed limits per column from the truth, the measurements, the reference and the bounds of the whole run
        # (the window only slides in time), with room at the top for the metrics; wider forecast bands are clipped
        lim = lambda vals: (min(np.nanmin(v) for v in vals), max(np.nanmax(v) for v in vals))
        X = [runs[m]["x"][:, 0] for m in methods] + [np.array([float(c["X_min"]), float(c.get("X_max") or c["X_min"])])]
        P = [runs[m]["x"][:, 2] for m in methods] + [runs[m]["y"] for m in methods] + [runs[methods[0]]["r"]]
        MU = [mu[m][0] for m in methods] + [mu[m][2] for m in methods]
        for col, vals in enumerate((X, P, MU)):
            lo, hi = lim(vals)
            pad = 0.1 * (hi - lo)
            ax[0, col].set_ylim(lo - pad, hi + 2.6 * pad)
        for a in ax.ravel():
            a.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        for a in ax[-1]:
            a.set_xlabel("time (h)")
            a.xaxis.set_major_locator(plt.MultipleLocator(12))
        handles = [plt.Line2D([], [], **true_st, label="true"),
                   plt.Line2D([], [], **{**meas_st, "ms": 2.0 * meas_st["ms"]}, label="measured"),
                   plt.matplotlib.patches.Patch(**{k: v for k, v in band_kw.items() if k != "zorder"},
                                                label=f"forecast, next {L * t_s:g} h (95%)"),
                   plt.Line2D([], [], color=st["color"], **part_st, label="particles"),
                   plt.Line2D([], [], **bound_st, label="bounds"),
                   plt.Line2D([], [], **ref_st, label="reference"),
                   plt.Line2D([], [], **{**viol_st, "ms": 1.3 * viol_st["ms"]}, label="bound violated")]
        h = fig.get_figheight()
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.5 / h, top=1 - 0.8 / h, hspace=0.12, wspace=0.22)
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.6, handletextpad=0.4, columnspacing=1.0)
        fig.text(0.5, 1 - 0.36 / h, "running averages up to the current time:  CRPS and tracking RMSE in "
                 r"$10^{-2}$ g/L,  CRPS of $\mu$ in $10^{-3}$ 1/h", ha="center", va="top", fontsize=8, color="#555555")

        def lin(y, k, f):                                    # the values up to sample k, then part of the next step
            if f == 0 or k + 1 >= len(y):
                return y[:k + 1]
            return np.concatenate([y[:k + 1], [y[k] + f * (y[k + 1] - y[k])]], axis=0)

        def frame(k, f):
            tc = t[k] + f * t_s                                       # current time, between samples k and k + 1
            th = t[k] + t_s * np.arange(L + 1)
            tt = lin(t, k, f)
            right = max(window, tc + t_s * L + 2)
            for row, m in enumerate(methods):
                r, A, M = runs[m], art[m], met[m]
                for col, sx in ((0, 0), (1, 2)):
                    mean, q = fc[m][sx]
                    A[f"band{col}"].remove()
                    A[f"band{col}"] = ax[row, col].fill_between(th, q[0, k], q[1, k], **band_kw)
                    for i, ln in enumerate(A[f"fan{col}"]):
                        ln.set_data(th, mean[k, i])
                    A[f"true{col}"].set_data(tt, lin(r["x"][:, sx], k, f))
                A["meas"].set_data(t[:k + 1], r["y"][:k + 1])
                mu_p, _, mu_t = mu[m]
                part = lin(mu_p, k, f)
                for i, ln in enumerate(A["mu"]):
                    ln.set_data(tt, part[:, i])
                A["true2"].set_data(tt, lin(mu_t, k, f))
                v = np.nonzero(np.asarray(r["violation"][:k + 1], bool))[0]
                A["viol"].set_data(t[v], np.full(len(v), pp.VIOLATION_Y))
                A["text"][0].set_text(f"violations  {int(M['violations'][k])}     CRPS  {_fmt(M['X'][k], 1e2)}")
                A["text"][1].set_text(f"CRPS  {_fmt(M['P'][k], 1e2)}     tracking RMSE  {_fmt(M['rmse'][k], 1e2)}")
                A["text"][2].set_text(f"CRPS  {_fmt(M['mu'][k], 1e3)}")
                for ln in A["now"]:
                    ln.set_xdata([tc, tc])
            ax[0, 0].set_xlim(right - window, right)                    # shared x

        steps = [(k, j / sub) for k in range(T - 1) for j in range(sub)] + [(T - 1, 0.0)]
        frames = []                                       # full-colour frames
        for k, f in steps:
            frame(k, f)
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = [round(1000 / fps)] * (len(frames) - 1) + [2000]          # hold the last frame for 2 s
    if out.suffix.lower() == ".webp":
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration, loop=0,
                       quality=quality, method=4)
    else:
        save_gif(frames, out, duration, colors)
    return out, met


def save_gif(frames, path, duration, colors=48):
    """GIF with one palette of `colors` colors for all frames, no dithering. The palette is learned from frames
    spread over the run plus swatches of every style colour, so that small marks (the measurements) keep their
    colour; the figure has few distinct colours, so the file is much smaller than with 256 colours per frame."""
    W, H = frames[0].size
    picks = np.linspace(0, len(frames) - 1, 4).astype(int)
    sample = Image.new("RGB", (W, H * len(picks) + 40), "white")
    for j, k in enumerate(picks):
        sample.paste(frames[k], (0, H * j))
    style = [pp.STYLE["true"]["color"], pp.MEASURED["color"], pp.FILTER_STYLE["color"], pp.FILTER_STYLE["fill"],
             pp.BOUND["color"], pp.REFERENCE["color"], pp.VIOLATION_MARKER["color"]]
    sw = W // len(style)
    for j, c in enumerate(style):
        sample.paste(tuple(int(255 * v) for v in to_rgb(c)), (j * sw, H * len(picks), (j + 1) * sw, H * len(picks) + 40))
    pal = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=duration, loop=0, optimize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--sub", type=int, default=2, help="frames per sampling interval (1: one frame per sample)")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--quality", type=int, default=80, help="WebP quality")
    ap.add_argument("--colors", type=int, default=48, help="GIF: size of the palette shared by all frames")
    ap.add_argument("--window", type=float, default=60.0, help="time window in hours, following the current time")
    ap.add_argument("--out", default=str(ROOT / "assets" / "closed_loop.webp"))
    a = ap.parse_args()
    cfg = load_config(CONFIG_DIR / "bioreactor.yaml")
    setup, runs = run_cached(cfg, a.methods, a.seed)
    out, met = animate(runs, setup, a.methods, a.out, fps=a.fps, sub=a.sub, dpi=a.dpi, window=a.window,
                       colors=a.colors, quality=a.quality)
    for m in a.methods:                                       # the last frame shows the run's metrics
        ref = bio_summary(runs[m], setup)
        last = {k: v[-1] for k, v in met[m].items()}
        assert np.isclose(last["X"], ref["CRPS X"]) and np.isclose(last["P"], ref["CRPS P"]), m
        assert np.isclose(last["mu"], ref["CRPS mu"]) and np.isclose(last["rmse"], ref["tracking RMSE"]), m
        assert last["violations"] == ref["washouts"], m
    print(f"written {out} ({out.stat().st_size / 1e6:.1f} MB); last-frame metrics equal bio_summary")


if __name__ == "__main__":
    main()

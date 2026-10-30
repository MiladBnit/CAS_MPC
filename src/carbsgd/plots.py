"""Figures of the bioreactor case study, in the style of the paper (L4DC, PMLR
format): the closed loop of one seed (plot_bio_paper) and the Monte Carlo over
seeds (plot_mc_paper), and save_figure.

The style settings below are module globals read when a figure is made, so a
notebook can override them before plotting, e.g.

    import carbsgd.plots as pp
    pp.BAND_ALPHA = 0.5; pp.FILTER_STYLE["color"] = "#0A8564"; pp.DPI = 300
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Colour by role (Okabe-Ito palette, colour-blind safe): the filters blue, the measurements orange,
# the truth black, the bounds and the violations red, the reference dark grey.
# STYLE: one colour per method, used when ONE_FILTER_COLOR is False.
STYLE = {
    "true":  dict(color="#1D1D1D", ls="-", lw=0.55, zorder=10),
    "ca":    dict(color="#0A8564", fill="#41AC4176", ls="-", lw=0.9, zorder=9, fill_zorder=5, fill_alpha=0.35),
    "param": dict(color="#128ADA", fill="#0795F380", ls="-", lw=0.9, zorder=8, fill_zorder=4, fill_alpha=0.35),
    "rbpf":  dict(color="#E96130", fill="#D89A5C76", ls="-", lw=0.9, zorder=7, fill_zorder=3, fill_alpha=0.5),
    "ce":    dict(color="#9254A3", fill="#7E3F8F50", ls="-", lw=0.9, zorder=6, fill_zorder=2, fill_alpha=0.25),
    "ca0":   dict(color="#5A9E8C", fill="#9FD3C476", ls="-", lw=0.9, zorder=9, fill_zorder=5, fill_alpha=0.35),
}
ROW_PAPER = {"ca": "control-aware\nkernel", "param": "parameter\nkernel", "rbpf": "RBPF", "ce": "CE-MPC",
             "ca0": "control-aware\n" + r"kernel, $\omega = 0$"}
SHORT_PAPER = {"ca": "CA", "param": "Par.", "rbpf": "PF", "ce": "CE", "ca0": r"CA$_0$"}
BOUND = dict(color="#D62728", ls=(0, (4, 2)), lw=1.0, zorder=12)          # bounds X_min, X_max: red, on top
REFERENCE = dict(color="#4D4D4D", ls=(0, (4, 2)), lw=1.0, zorder=12)      # reference of P: dark grey, on top
BAND_ALPHA = 0.6          # the 95% interval, x the method's fill_alpha (the particles drawn above it)
PARTICLES = dict(lw=0.45, alpha=0.55)
MEASURED = dict(color="#FF7300", ls="none", marker="o", ms=1.25, mew=0, zorder=11)  # measurements of P: orange
ONE_FILTER_COLOR = True    # closed-loop figure: every method in FILTER_STYLE (the row labels name them);
                           # False: each method in its STYLE colour
FILTER_STYLE = dict(color="#0072B2", fill="#56B4E9", ls="-", lw=0.9, zorder=9, fill_zorder=5, fill_alpha=0.5)   # blue
# Monte Carlo figure: the proposed method in the filter blue, the others neutral grey
MC_STYLE = {"ca": dict(color="#0072B2", fill="#56B4E9", fill_alpha=0.55),
            "param": dict(color="#6E6E6E", fill="#BDBDBD", fill_alpha=0.55),
            "rbpf": dict(color="#6E6E6E", fill="#BDBDBD", fill_alpha=0.55),
            "ce": dict(color="#6E6E6E", fill="#BDBDBD", fill_alpha=0.55),
            "ca0": dict(color="#6E6E6E", fill="#BDBDBD", fill_alpha=0.55)}
SHOW_MEAN = False          # the forecast mean (and the particles' mean of mu)
SHOW_MEASUREMENTS = True   # the measurements of P (X is not measured)
LEGEND_NCOL = None         # closed-loop legend: columns (None: one row); e.g. 4 for two rows
LEGEND_TOP = 0.93          # top of the panels (the legend above it); lower it for a two-row legend
VIOLATION = "marker"       # steps outside the bounds: "marker" (on the time axis) or "shade" (WASHOUT bands)
VIOLATION_MARKER = dict(marker="^", color="#D62728", ms=3.0, mew=0, ls="none", zorder=13)   # red triangles
VIOLATION_Y = 0.04         # height of the markers in the panel (0 bottom, 1 top)
WASHOUT = dict(color="#D62728", alpha=0.18, lw=0, zorder=1)   # the shading, VIOLATION "shade"
DPI = 600                 # paper figures: drawn and saved at this resolution
PAPER_RC = {
    "font.family": ["Times New Roman", "DejaVu Serif"],   # a named family (not "serif"), resolved when drawn
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.0, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.3,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.bbox": "tight",
}
TEXT_WIDTH = 6.5          # inches (PMLR / L4DC text width)


def _rc():
    """The paper style, with the current DPI."""
    return {**PAPER_RC, "figure.dpi": DPI, "savefig.dpi": DPI}


def save_figure(fig, stem, formats=("pdf",), dpi=None):
    """stem.pdf (vector; any rasterized part at dpi), and the other formats given
    (e.g. "png"), white background. Returns the paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = stem.with_suffix("." + fmt)
        fig.savefig(path, format=fmt, dpi=dpi or DPI, bbox_inches="tight", facecolor="white")
        paths.append(path)
    return paths


def _washout_spans(ax, t, violated, t_s):
    """Shade the steps at which the plant was outside the constraint set."""
    v = np.asarray(violated, bool)
    k = 0
    while k < len(v):
        if v[k]:
            j = k
            while j + 1 < len(v) and v[j + 1]:
                j += 1
            ax.axvspan(t[k] - t_s / 2, t[j] + t_s / 2, **WASHOUT)
            k = j + 1
        else:
            k += 1


def _band(ax, t, mean, lo, hi, samples, st, show_samples=True, show_mean=True):
    """Light interval (if lo is not None), the particles above it, and the mean line."""
    if lo is not None:
        ax.fill_between(t, lo, hi, color=st["fill"], alpha=BAND_ALPHA * st["fill_alpha"], lw=0,
                        zorder=st["fill_zorder"])
    if show_samples:
        ax.plot(t, samples, color=st["color"], **PARTICLES, zorder=st["fill_zorder"] + 1)
    if show_mean:
        ax.plot(t, mean, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["zorder"])


def plot_bio_paper(runs, setup, methods=("ca", "ca0", "param", "rbpf"), level=0.95, figsize=None, show_samples=True,
                   show_mean=None, show_measurements=None):
    """Closed loop for the paper: one row per method; columns biomass X with the
    bounds, product P with the reference, and growth rate mu at the true state.
    X and P: the controller's one-step forecast, a Gaussian mixture over the
    scenarios, as its central `level` interval (mixture quantiles), the scenario
    means (show_samples) and its mean (show_mean); mu: the particles. The truth is
    a thin black line, the measurements of P dots (show_measurements), the steps
    outside the bounds red markers on the time axis or shaded bands (VIOLATION).
    show_mean, show_measurements: None reads SHOW_MEAN, SHOW_MEASUREMENTS."""
    show_mean = SHOW_MEAN if show_mean is None else show_mean
    show_measurements = SHOW_MEASUREMENTS if show_measurements is None else show_measurements
    from .bio_setup import growth_tracking
    from .metrics import mixture_quantiles
    a_lo, a_hi = 0.5 * (1 - level), 0.5 * (1 + level)
    c = setup.cfg["bio"]
    t_s = float(c["t_s"])
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(len(methods), 3, sharex=True, sharey="col",
                               figsize=figsize or (TEXT_WIDTH, 1.03 * len(methods) + 0.4))
        ax = np.atleast_2d(ax)
        for row, m in enumerate(methods):
            r, st = runs[m], (FILTER_STYLE if ONE_FILTER_COLOR else STYLE[m])
            T = len(r["y"])
            t = np.arange(T) * t_s
            ws = r["w"] / r["w"].sum(axis=1, keepdims=True)
            mean_f, sd_f = r["pred"][:-1, :, 0, :], r["pred_std"][:-1, :, 0, :]   # forecast of t, made at t - 1
            for col, k in ((0, 0), (1, 2)):
                a = ax[row, col]
                q = mixture_quantiles(mean_f[..., k], sd_f[..., k], ws[:-1], (a_lo, a_hi))
                _band(a, t[1:], np.einsum("tn,tn->t", ws[:-1], mean_f[..., k]), q[0], q[1],
                      mean_f[..., k], st, show_samples, show_mean)
                a.plot(t, r["x"][:, k], **STYLE["true"])
            a = ax[row, 0]
            a.axhline(float(c["X_min"]), **BOUND)
            if c.get("X_max") is not None:
                a.axhline(float(c["X_max"]), **BOUND)
            if VIOLATION == "shade":
                _washout_spans(a, t, r["violation"], t_s)
            else:                                     # one marker per step outside the bounds, on the time axis
                v = np.asarray(r["violation"], bool)
                mk, = a.plot(t[v], np.full(v.sum(), VIOLATION_Y), transform=a.get_xaxis_transform(), clip_on=False,
                             **VIOLATION_MARKER)
                mk.set_in_layout(False)                   # inside the panel: not a decoration for tight_layout
            ax[row, 1].step(t, r["r"], where="post", **REFERENCE)
            if show_measurements:
                ax[row, 1].plot(t, r["y"], **MEASURED)
            mu, mu_mean, mu_t = growth_tracking(r, setup)
            _band(ax[row, 2], t, mu_mean, None, None, mu, st, show_samples, show_mean)   # mu: the particles
            ax[row, 2].plot(t, mu_t, **STYLE["true"])
            ax[row, 0].set_ylabel(ROW_PAPER.get(m, m), color="#1D1D1D" if ONE_FILTER_COLOR else st["color"],
                                  fontsize=8, linespacing=1.1)
        ax[0, 0].set_title(r"biomass $X$ (g/L)")
        ax[0, 1].set_title(r"product $P$ (g/L)")
        ax[0, 2].set_title(r"growth rate $\mu$ (1/h)")
        for a in ax.ravel():
            a.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        T_h = len(runs[methods[0]]["y"]) * t_s
        for a in ax[-1]:
            a.set_xlabel("time (h)")
            a.set_xlim(0, T_h)
            a.set_xticks(np.arange(0, T_h + 1, 48))
        gray = FILTER_STYLE["color"] if ONE_FILTER_COLOR else "#6E6E6E"
        handles = [plt.Line2D([], [], **STYLE["true"], label="true")]
        if show_measurements:
            handles.append(plt.Line2D([], [], **{**MEASURED, "ms": 2.5 * MEASURED.get("ms", 1.6)}, label="measured"))
        if show_mean:
            handles.append(plt.Line2D([], [], color=gray, lw=0.9, label="forecast mean"))
        band = (dict(color=FILTER_STYLE["fill"], alpha=BAND_ALPHA * FILTER_STYLE["fill_alpha"]) if ONE_FILTER_COLOR
                else dict(color=gray, alpha=0.2))
        handles += [plt.matplotlib.patches.Patch(**band, lw=0, label=f"{int(100 * level)}% interval"),
                   plt.Line2D([], [], color=gray, **PARTICLES, label="particles"),
                   plt.Line2D([], [], **BOUND, label="bounds"),
                   plt.Line2D([], [], **REFERENCE, label="reference"),
                   (plt.matplotlib.patches.Patch(**WASHOUT, label="bound violated") if VIOLATION == "shade"
                    else plt.Line2D([], [], **{**VIOLATION_MARKER, "ms": 1.4 * VIOLATION_MARKER["ms"]},
                                    label="bound violated"))]
        fig.tight_layout(rect=(0, 0, 1, LEGEND_TOP), h_pad=0.3, w_pad=0.8)   # the panels first, then the legend above
        fig.legend(handles=handles, loc="upper center", ncol=LEGEND_NCOL or len(handles), bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.3, handletextpad=0.4, columnspacing=0.9)
    return fig


def plot_mc_paper(df, figsize=None, washout_methods=("ca", "ca0", "param", "rbpf", "ce"),
                  methods=("ca", "ca0", "param", "rbpf")):
    """Monte Carlo for the paper, one row of box plots (one point per seed):
    washouts (CE-MPC included), tracking RMSE and the CRPS of X, P and mu."""
    panels = [("washouts", "washouts (steps)", washout_methods, 1.0),
              ("tracking RMSE", r"tracking RMSE ($10^{-2}$ g/L)", methods, 100.0),
              ("CRPS X", r"CRPS $X$ ($10^{-2}$ g/L)", methods, 100.0),
              ("CRPS P", r"CRPS $P$ ($10^{-2}$ g/L)", methods, 100.0),
              ("CRPS mu", r"CRPS $\mu$ ($10^{-3}$ 1/h)", methods, 1000.0)]
    rng = np.random.default_rng(0)
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(1, len(panels), figsize=figsize or (TEXT_WIDTH, 1.75),
                               gridspec_kw=dict(width_ratios=[len(p[2]) for p in panels]))
        for a, (metric, title, ms, scale) in zip(ax, panels):
            data = [scale * df.loc[df["method"] == m, metric].to_numpy(float) for m in ms]
            bp = a.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                           medianprops=dict(lw=1.0), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                           boxprops=dict(lw=0.6))
            for k, m in enumerate(ms):
                st = MC_STYLE.get(m, STYLE[m])
                bp["boxes"][k].set(facecolor=plt.matplotlib.colors.to_rgba(st["fill"], st["fill_alpha"]),
                                   edgecolor=st["color"])
                bp["medians"][k].set(color=st["color"])
                for part in ("whiskers", "caps"):
                    for line in bp[part][2 * k:2 * k + 2]:
                        line.set(color=st["color"])
                a.plot(k + 1 + 0.10 * rng.standard_normal(len(data[k])), data[k], "o", ms=1.8,
                       color=st["color"], alpha=0.75, mew=0, zorder=5)
            a.set_xticks(range(1, len(ms) + 1), [SHORT_PAPER.get(m, m) for m in ms])
            a.tick_params(axis="x", length=0)
            a.set_title(title, fontsize=8)
            a.grid(True, axis="y")
            a.grid(False, axis="x")
        fig.tight_layout(w_pad=0.5)
    return fig

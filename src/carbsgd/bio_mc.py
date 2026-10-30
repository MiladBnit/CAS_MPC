"""Monte Carlo of the bioreactor case study: closed-loop runs of every method over a
set of seeds (in parallel processes), the metrics of bio_summary per run, their mean
and spread over the seeds, and the results saved with the configuration used.

The seed sets the plant and measurement noise, the initial particles and RBPF's
random numbers (closed_loop.run_closed_loop); the configuration is the same for
every run.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# the metrics of bio_summary, in reporting order (those present are reported)
METRICS = ["washouts", "washouts low", "washouts high", "tracking RMSE", "CRPS X", "CRPS P", "CRPS mu", "computation ms"]


def _one_run(cfg, method, seed):
    """One closed-loop run (in a worker process): the metrics of bio_summary."""
    from .bio_setup import bio_summary
    from .closed_loop import make_setup, run_closed_loop
    setup = make_setup(cfg, [method])
    res = run_closed_loop(setup, method, seed)
    return {"method": method, "seed": seed, **bio_summary(res, setup)}


def run_mc(cfg, methods, seeds, workers=None, progress=True):
    """All (method, seed) runs, in parallel processes. Returns one row per run."""
    df = run_mc_jobs(cfg, [(m, s) for s in seeds for m in methods], workers, progress)
    df["method"] = pd.Categorical(df["method"], categories=list(methods), ordered=True)
    return df.sort_values(["method", "seed"]).reset_index(drop=True)


def mc_table(df, digits=4):
    """Mean +- standard deviation over the seeds, per method and metric."""
    cols = [c for c in METRICS if c in df.columns]
    g = df.groupby("method", observed=True)[cols]
    mean, std = g.mean(), g.std(ddof=1)
    fmt = lambda m, s: f"{m:.{digits}g} ± {s:.{digits}g}" if np.isfinite(s) else f"{m:.{digits}g}"
    return pd.DataFrame({c: [fmt(mean.loc[i, c], std.loc[i, c]) for i in mean.index] for c in cols},
                        index=mean.index).T


def save_mc(df, cfg, tag, root=None):
    """results/bioreactor_mc_<tag>/: runs.csv (one row per run) and config.yaml (the
    configuration used). Returns the directory."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2] / "results"
    out = root / f"bioreactor_mc_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(cfg), f, sort_keys=False)
    return out


def _plain(v):
    """Configuration as plain Python types (for YAML)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------- saved results, reused
# The results are reused when the configuration is the same (compared after a YAML round
# trip). The code itself is not part of the comparison: after a change in the code, run
# again (rerun=True).

def _results_dir():
    return Path(__file__).resolve().parents[2] / "results"


def _same_config(a, b):
    import json
    canon = lambda c: json.dumps(yaml.safe_load(yaml.safe_dump(_plain(c), sort_keys=True)), sort_keys=True)
    return canon(a) == canon(b)


def config_hash(cfg):
    """Short fingerprint of the configuration."""
    import hashlib
    import json
    text = json.dumps(yaml.safe_load(yaml.safe_dump(_plain(cfg), sort_keys=True)), sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def run_cached(cfg, methods, seed, rerun=False, progress=True):
    """Closed-loop runs of the methods for one seed, saved in
    results/bioreactor_runs/<config hash>/ (<method>_seed<seed>.pkl, and the
    configuration in config.yaml) and loaded from there when they exist (unless
    rerun). Returns (setup, runs); the setup is built for the methods that run
    (for the plots alone, without filters and controllers)."""
    import pickle
    from .closed_loop import make_setup, run_closed_loop
    out = _results_dir() / "bioreactor_runs" / config_hash(cfg)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "config.yaml").exists():
        with open(out / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(_plain(cfg), f, sort_keys=False)
    files = {m: out / f"{m}_seed{int(seed)}.pkl" for m in methods}
    todo = [m for m in methods if rerun or not files[m].exists()]
    setup = make_setup(cfg, todo)
    runs = {}
    for m in methods:
        if m in todo:
            t0 = time.perf_counter()
            runs[m] = run_closed_loop(setup, m, int(seed))
            with open(files[m], "wb") as f:
                pickle.dump(runs[m], f)
            msg = f"ran in {time.perf_counter() - t0:.0f} s"
        else:
            with open(files[m], "rb") as f:
                runs[m] = pickle.load(f)
            msg = "loaded"
        if progress:
            print(f"{m:6s} {msg}")
    if progress:
        print(f"results: {out.relative_to(_results_dir().parent)}")
    return setup, runs


def run_mc_cached(cfg, methods, seeds, tag, workers=None, rerun=False, progress=True):
    """run_mc with the results of results/bioreactor_mc_<tag>/ reused: if they were
    made with the same configuration, the saved (method, seed) runs are loaded and
    only the missing ones run (then all are saved again). A different configuration
    under the same tag raises (choose another tag, or rerun=True to replace them).
    Returns (df, directory)."""
    out = _results_dir() / f"bioreactor_mc_{tag}"
    old = None
    if not rerun and (out / "runs.csv").exists() and (out / "config.yaml").exists():
        with open(out / "config.yaml", encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        if not _same_config(saved, cfg):
            raise ValueError(f"{out} holds results of a different configuration: use another tag, "
                             f"or rerun=True to replace them")
        old = pd.read_csv(out / "runs.csv")
    want = {(m, int(s)) for m in methods for s in seeds}
    have = set() if old is None else set(zip(old["method"], old["seed"].astype(int)))
    missing = sorted(want - have, key=lambda j: (list(methods).index(j[0]), j[1]))
    if progress:
        print(f"{len(want & have)} runs loaded, {len(missing)} to run")
    parts = [] if old is None else [old[old.apply(lambda r: (r["method"], int(r["seed"])) in want, axis=1)]]
    if missing:
        parts.append(run_mc_jobs(cfg, missing, workers, progress))
    df = pd.concat(parts, ignore_index=True)
    df["method"] = pd.Categorical(df["method"], categories=list(methods), ordered=True)
    df = df.sort_values(["method", "seed"]).reset_index(drop=True)
    if missing or old is None:
        save_mc(df, cfg, tag)
    return df, out


def run_mc_jobs(cfg, jobs, workers=None, progress=True):
    """The given (method, seed) runs, in parallel processes. One row per run."""
    jobs = [(m, int(s)) for m, s in jobs]
    workers = int(workers or min(len(jobs), os.cpu_count() or 1))
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_one_run, cfg, m, s) for m, s in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if progress:
                print(f"\r{k}/{len(jobs)} runs ({time.perf_counter() - t0:.0f} s)", end="", flush=True)
    if progress:
        print()
    return pd.DataFrame(rows)

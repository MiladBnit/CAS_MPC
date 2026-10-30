"""Loading and saving of the configuration files in config/."""

from __future__ import annotations

import copy
import datetime
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load config/default.yaml, then an optional override file and dict."""
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != (CONFIG_DIR / "default.yaml").resolve():
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def parse_overrides(items: list[str] | None) -> dict:
    """["filter.epsilon=1e-3", "closed_loop.methods=[ce, ca]"] -> nested dict (YAML values)."""
    out: dict = {}
    for item in items or []:
        key, val = item.split("=", 1)
        d = out
        *parents, leaf = key.strip().split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(val)
    return out


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_config(cfg: dict, out_dir: str | Path, name: str = "config_used.yaml") -> Path:
    """Save the config used for a result, with the git hash and a timestamp."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_hash": _git_hash(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path = out_dir / name
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"_meta": meta, **cfg}, f, sort_keys=False)
    return path


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path

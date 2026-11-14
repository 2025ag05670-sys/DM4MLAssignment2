"""Shared helpers: configuration, logging, paths and I/O.

Every stage module exposes a ``run(cfg)`` function and uses these helpers so
that paths, logging and seeds are handled identically everywhere.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


# ---------------------------------------------------------------- config ----
def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the YAML config and resolve every entry under ``paths`` to an absolute path."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    cfg["paths"] = {k: str(PROJECT_ROOT / v) for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return cfg


def path(cfg: dict, key: str, *parts: str) -> Path:
    """Build a path inside one of the configured directories."""
    p = Path(cfg["paths"][key]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------- logging ----
_LOGGER_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to stdout and to logs/pipeline.log."""
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        root = logging.getLogger("pipeline")
        root.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(sh)
        root.addHandler(fh)
        root.propagate = False
        _LOGGER_CONFIGURED = True
    return logging.getLogger(f"pipeline.{name}")


@contextmanager
def timed(logger: logging.Logger, label: str):
    start = time.perf_counter()
    logger.info("START  %s", label)
    yield
    logger.info("FINISH %s (%.1fs)", label, time.perf_counter() - start)


# ------------------------------------------------------------------- I/O ----
def write_parquet(df: pd.DataFrame, p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def read_parquet(p: Path) -> pd.DataFrame:
    if not Path(p).exists():
        raise FileNotFoundError(f"{p} not found - run the upstream pipeline stage first.")
    return pd.read_parquet(p)


def write_json(obj: Any, p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return p


def read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Small dependency-free markdown table renderer."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

"""Stage 1 - Data ingestion.

* Downloads MovieLens-1M (users, items, interactions) from the configured
  source, with retries and a local-cache fallback.
* Normalises column names into a common raw schema.
* Optionally injects a small, seeded, documented set of data-quality anomalies
  (see ``configs/config.yaml -> ingestion.anomaly_injection``) so that the
  validation stage is exercised on realistic dirty data.
* Writes ``data/raw/{users,items,interactions}.csv`` (everything kept as text,
  exactly as "received") plus an ingestion manifest with row counts and
  MD5 checksums for lineage / reproducibility.
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger, md5, path, set_seed, timed, write_json

log = get_logger("ingestion")

RAW_COLUMNS = {
    "users": ["user_id", "gender", "age", "occupation", "zipcode"],
    "items": ["movie_id", "title", "genres"],
    "interactions": ["user_id", "movie_id", "rating", "timestamp"],
}


# ------------------------------------------------------------ download ----
def _download(url: str, dest: Path, retries: int, timeout: int) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cache hit  %s", dest.name)
        return dest
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            log.info("download   %s (attempt %d/%d)", url, attempt, retries)
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                dest.write_bytes(resp.read())
            return dest
        except Exception as err:  # noqa: BLE001 - we re-raise after retries
            last_err = err
            log.warning("download failed: %s", err)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_err}")


def _read_github_csv(cfg: dict) -> dict[str, pd.DataFrame]:
    icfg = cfg["ingestion"]
    gcfg = icfg["github_csv"]
    frames = {}
    for table, fname in gcfg["files"].items():
        dest = path(cfg, "raw_source", fname)
        _download(f"{gcfg['base_url']}/{fname}", dest,
                  icfg["download_retries"], icfg["timeout_seconds"])
        df = pd.read_csv(dest, sep=gcfg["sep"], encoding=gcfg["encoding"], dtype=str,
                         keep_default_na=False)
        df = df.loc[:, [c for c in df.columns if c in RAW_COLUMNS[table]]]
        frames[table] = df[RAW_COLUMNS[table]]
    return frames


DAT_FILES = {"users": "users.dat", "items": "movies.dat", "interactions": "ratings.dat"}


def _parse_dat(text: str, table: str) -> pd.DataFrame:
    """Parse a MovieLens-1M '::'-separated .dat file into the raw text schema."""
    cols = RAW_COLUMNS[table]
    rows = [line.split("::")[: len(cols)] for line in text.splitlines() if line.strip()]
    return pd.DataFrame(rows, columns=cols, dtype=str)


def _read_kaggle_input(cfg: dict) -> dict[str, pd.DataFrame]:
    """Use a MovieLens-1M dataset attached to the Kaggle notebook (Add Input), if present."""
    root = Path(cfg["ingestion"]["kaggle_input"]["search_root"])
    hits = sorted(root.glob("**/ratings.dat")) if root.exists() else []
    for hit in hits:
        folder = hit.parent
        if all((folder / f).exists() for f in DAT_FILES.values()):
            log.info("using attached Kaggle dataset at %s", folder)
            return {t: _parse_dat((folder / f).read_text(encoding="latin-1"), t) for t, f in DAT_FILES.items()}
    raise FileNotFoundError(f"no users.dat/movies.dat/ratings.dat found under {root}")


def _read_grouplens_zip(cfg: dict) -> dict[str, pd.DataFrame]:
    icfg = cfg["ingestion"]
    dest = path(cfg, "raw_source", "ml-1m.zip")
    _download(icfg["grouplens_zip"]["url"], dest, icfg["download_retries"], icfg["timeout_seconds"])
    with zipfile.ZipFile(dest) as zf:
        return {t: _parse_dat(zf.read(f"ml-1m/{f}").decode("latin-1"), t) for t, f in DAT_FILES.items()}


def _read_local(cfg: dict) -> dict[str, pd.DataFrame]:
    d = Path(cfg["ingestion"]["local"]["dir"])
    gcfg = cfg["ingestion"]["github_csv"]
    frames = {}
    for table, fname in gcfg["files"].items():
        df = pd.read_csv(d / fname, sep=gcfg["sep"], encoding=gcfg["encoding"], dtype=str,
                         keep_default_na=False)
        frames[table] = df[RAW_COLUMNS[table]]
    return frames


def _read_auto(cfg: dict) -> dict[str, pd.DataFrame]:
    """Try each configured source in order and record which one succeeded."""
    errors = {}
    for name in cfg["ingestion"]["auto_order"]:
        try:
            frames = READERS[name](cfg)
            cfg["ingestion"]["_resolved_source"] = name
            log.info("data source resolved: %s", name)
            return frames
        except Exception as err:  # noqa: BLE001 - fall through to the next source
            errors[name] = repr(err)
            log.warning("source %s unavailable: %s", name, err)
    raise RuntimeError(f"No MovieLens source could be read: {errors}")


READERS = {"github_csv": _read_github_csv, "grouplens_zip": _read_grouplens_zip,
           "kaggle_input": _read_kaggle_input, "local": _read_local, "auto": _read_auto}


# ------------------------------------------------------ anomaly injection ----
def inject_anomalies(frames: dict[str, pd.DataFrame], spec: dict, seed: int):
    """Inject documented data-quality problems. Returns (frames, manifest)."""
    rng = np.random.default_rng(seed)
    manifest: dict[str, dict[str, int]] = {}

    def pick(df: pd.DataFrame, n: int) -> np.ndarray:
        return rng.choice(df.index.to_numpy(), size=min(n, len(df)), replace=False)

    # ---- interactions ----
    it = frames["interactions"].copy()
    s = spec["interactions"]
    extra = []
    extra.append(it.loc[pick(it, s["exact_duplicates"])])                       # exact dups
    rr = it.loc[pick(it, s["re_ratings"])].copy()                                # re-ratings
    rr["timestamp"] = (rr["timestamp"].astype(int) + rng.integers(3600, 86400 * 30, len(rr))).astype(str)
    rr["rating"] = rng.integers(1, 6, len(rr)).astype(str)
    extra.append(rr)
    idx = pick(it, s["missing_rating"]);       it.loc[idx, "rating"] = ""
    idx = pick(it, s["missing_user_id"]);      it.loc[idx, "user_id"] = ""
    idx = pick(it, s["out_of_range_rating"]);  it.loc[idx, "rating"] = rng.choice(["0", "6", "7.5", "-1", "10"], len(idx))
    idx = pick(it, s["non_numeric_rating"]);   it.loc[idx, "rating"] = rng.choice(["five", "abc", "4 stars", "?"], len(idx))
    idx = pick(it, s["invalid_timestamp"]);    it.loc[idx, "timestamp"] = rng.choice(
        ["2081376000", "-86400", "2001-13-45", "not_a_date", "0"], len(idx))
    idx = pick(it, s["orphan_user"]);          it.loc[idx, "user_id"] = rng.integers(90000, 99999, len(idx)).astype(str)
    idx = pick(it, s["orphan_item"]);          it.loc[idx, "movie_id"] = rng.integers(900000, 999999, len(idx)).astype(str)
    it = pd.concat([it, *extra]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    manifest["interactions"] = {k: int(v) for k, v in s.items()}

    # ---- users ----
    us = frames["users"].copy()
    s = spec["users"]
    dups = us.loc[pick(us, s["exact_duplicates"])]
    idx = pick(us, s["missing_gender"]);   us.loc[idx, "gender"] = ""
    idx = pick(us, s["lowercase_gender"]); us.loc[idx, "gender"] = us.loc[idx, "gender"].str.lower()
    idx = pick(us, s["invalid_age"]);      us.loc[idx, "age"] = rng.choice(["200", "-5", "abc"], len(idx))
    idx = pick(us, s["missing_zipcode"]);  us.loc[idx, "zipcode"] = ""
    us = pd.concat([us, dups]).reset_index(drop=True)
    manifest["users"] = {k: int(v) for k, v in s.items()}

    # ---- items ----
    im = frames["items"].copy()
    s = spec["items"]
    dups = im.loc[pick(im, s["exact_duplicates"])]
    idx = pick(im, s["missing_genres"]); im.loc[idx, "genres"] = ""
    idx = pick(im, s["missing_title"]);  im.loc[idx, "title"] = ""
    idx = pick(im, s["padded_title"]);   im.loc[idx, "title"] = "  " + im.loc[idx, "title"] + "   "
    im = pd.concat([im, dups]).reset_index(drop=True)
    manifest["items"] = {k: int(v) for k, v in s.items()}

    return {"users": us, "items": im, "interactions": it}, manifest


# ------------------------------------------------------------------ run ----
def run(cfg: dict) -> dict:
    set_seed(cfg["project"]["seed"])
    icfg = cfg["ingestion"]
    with timed(log, "ingestion"):
        frames = READERS[icfg["source_type"]](cfg)
        source_counts = {t: len(df) for t, df in frames.items()}
        log.info("source rows: %s", source_counts)

        injected = {}
        if icfg["anomaly_injection"]["enabled"]:
            frames, injected = inject_anomalies(frames, icfg["anomaly_injection"],
                                                cfg["project"]["seed"])
            log.info("injected anomalies: %s",
                     {t: sum(v.values()) for t, v in injected.items()})

        outputs = {}
        for table, df in frames.items():
            p = path(cfg, "raw", f"{table}.csv")
            df.to_csv(p, index=False)
            outputs[table] = {"file": str(Path(p).relative_to(Path(cfg["paths"]["raw"]).parents[1])),
                              "rows": len(df), "columns": list(df.columns), "md5": md5(p)}
            log.info("wrote %-13s %8d rows -> %s", table, len(df), p)

        manifest = {
            "dataset": icfg["dataset"],
            "source_type": icfg["source_type"],
            "resolved_source": icfg.get("_resolved_source", icfg["source_type"]),
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_row_counts": source_counts,
            "anomaly_injection_enabled": icfg["anomaly_injection"]["enabled"],
            "injected_anomalies": injected,
            "outputs": outputs,
        }
        write_json(manifest, path(cfg, "raw", "ingestion_manifest.json"))
    return manifest


if __name__ == "__main__":
    from src.utils import load_config
    run(load_config())

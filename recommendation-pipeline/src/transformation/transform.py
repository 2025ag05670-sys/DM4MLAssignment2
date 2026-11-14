"""Stage 4 - Data transformation.

* **Time-based train/test split** - a single global cut-off at the
  ``1 - test_fraction`` quantile of interaction time. Everything before the
  cut-off is training data, everything after is test data. This mimics
  production (train on the past, predict the future) and prevents leakage
  that a random split would cause.
* Test rows are restricted to *warm* users/items (seen in training with at
  least ``min_train_interactions_per_user`` interactions); cold-start rows are
  counted and reported.
* Derived datasets, all computed from **training data only**:
    - ``user_activity_summary.parquet``
    - ``item_popularity.parquet``
    - ``train_matrix.npz`` (sparse user x item rating matrix) + index maps
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils import get_logger, path, read_parquet, timed, write_json, write_parquet

log = get_logger("transformation")


def time_split(inter: pd.DataFrame, test_fraction: float, min_train: int):
    cutoff = inter.timestamp.quantile(1 - test_fraction)
    train = inter[inter.timestamp < cutoff].copy()
    test_all = inter[inter.timestamp >= cutoff].copy()
    counts = train.user_id.value_counts()
    warm_users = set(counts[counts >= min_train].index)
    warm_items = set(train.movie_id.unique())
    test = test_all[test_all.user_id.isin(warm_users) & test_all.movie_id.isin(warm_items)].copy()
    info = {
        "strategy": "global_time",
        "cutoff_utc": cutoff.isoformat(),
        "train_rows": len(train), "test_rows_before_filter": len(test_all), "test_rows": len(test),
        "train_period": [train.timestamp.min().isoformat(), train.timestamp.max().isoformat()],
        "test_period": [test_all.timestamp.min().isoformat(), test_all.timestamp.max().isoformat()],
        "cold_start_rows_dropped": int(len(test_all) - len(test)),
        "cold_start_users_in_test": int(len(set(test_all.user_id) - warm_users)),
        "train_users": int(train.user_id.nunique()), "train_items": int(train.movie_id.nunique()),
        "test_users": int(test.user_id.nunique()), "test_items": int(test.movie_id.nunique()),
    }
    return train, test, info


def user_activity_summary(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.merge(items[["movie_id", "genre_list"]], on="movie_id").explode("genre_list")
    n_genres = g.groupby("user_id").genre_list.nunique().rename("n_distinct_genres")
    s = train.groupby("user_id").agg(
        n_interactions=("rating", "size"), mean_rating=("rating", "mean"),
        std_rating=("rating", "std"), first_ts=("timestamp", "min"), last_ts=("timestamp", "max"))
    s["active_days"] = (s.last_ts - s.first_ts).dt.days + 1
    s["std_rating"] = s.std_rating.fillna(0.0)
    return s.join(n_genres).reset_index()


def item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    s = train.groupby("movie_id").agg(
        n_interactions=("rating", "size"), n_users=("user_id", "nunique"),
        mean_rating=("rating", "mean"), first_ts=("timestamp", "min"), last_ts=("timestamp", "max"))
    s["popularity_rank"] = s.n_interactions.rank(ascending=False, method="min").astype(int)
    s["popularity_pct"] = s.n_interactions / s.n_interactions.sum()
    return s.reset_index().sort_values("popularity_rank")


def interaction_matrix(train: pd.DataFrame):
    users = np.sort(train.user_id.unique())
    items = np.sort(train.movie_id.unique())
    uidx = pd.Series(np.arange(len(users)), index=users)
    iidx = pd.Series(np.arange(len(items)), index=items)
    m = sp.csr_matrix((train.rating.astype(np.float32).to_numpy(),
                       (uidx[train.user_id].to_numpy(), iidx[train.movie_id].to_numpy())),
                      shape=(len(users), len(items)))
    return m, pd.DataFrame({"user_id": users, "row": np.arange(len(users))}), \
        pd.DataFrame({"movie_id": items, "col": np.arange(len(items))})


def run(cfg: dict) -> dict:
    t = cfg["transformation"]
    with timed(log, "transformation"):
        inter = read_parquet(path(cfg, "processed", "interactions.parquet"))
        items = read_parquet(path(cfg, "processed", "items.parquet"))
        train, test, info = time_split(inter, t["test_fraction"], t["min_train_interactions_per_user"])
        test["relevant"] = test.rating >= t["relevance_threshold"]
        info["test_relevant_rows"] = int(test.relevant.sum())
        info["test_users_with_relevant"] = int(test[test.relevant].user_id.nunique())
        log.info("split: %s", {k: info[k] for k in ["cutoff_utc", "train_rows", "test_rows", "test_users"]})

        write_parquet(train, path(cfg, "processed", "train.parquet"))
        write_parquet(test, path(cfg, "processed", "test.parquet"))
        write_parquet(user_activity_summary(train, items), path(cfg, "processed", "user_activity_summary.parquet"))
        write_parquet(item_popularity(train), path(cfg, "processed", "item_popularity.parquet"))

        m, uix, iix = interaction_matrix(train)
        sp.save_npz(path(cfg, "processed", "train_matrix.npz"), m)
        write_parquet(uix, path(cfg, "processed", "user_index.parquet"))
        write_parquet(iix, path(cfg, "processed", "item_index.parquet"))
        info["matrix_shape"] = list(m.shape)
        info["matrix_density_pct"] = round(100 * m.nnz / (m.shape[0] * m.shape[1]), 3)
        log.info("interaction matrix %s density %.2f%%", m.shape, info["matrix_density_pct"])
        write_json(info, path(cfg, "reports", "split_info.json"))
    return info


if __name__ == "__main__":
    from src.utils import load_config
    run(load_config())

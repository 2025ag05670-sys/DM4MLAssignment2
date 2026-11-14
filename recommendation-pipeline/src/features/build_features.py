"""Stage 5 - Feature engineering & a light-weight feature store.

Features are computed **as of the train/test cutoff** from training data only,
so the exact same values are used for training and for offline prediction
(no leakage, no training/serving skew).

Offline store : ``data/features/{user,item,user_genre}_features.parquet``
Registry      : ``data/features/feature_registry.json`` (+ ``reports/feature_catalog.md``)
Access layer  : :class:`FeatureStore` - used by the models at training time and
                by ``src/serving/recommend.py`` at prediction time.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features.registry import FEATURES, as_records
from src.preparation.prepare import GENRES
from src.utils import (df_to_markdown, get_logger, path, read_json, read_parquet, timed,
                       write_json, write_parquet)

log = get_logger("features")
FEATURE_VERSION = "v1"


def genre_col(g: str) -> str:
    return "genre_" + g.lower().replace("'", "").replace("-", "_")


GENRE_COLS = [genre_col(g) for g in GENRES]


def build_item_features(train, items, cutoff, prior_m):
    gmean = train.rating.mean()
    s = train.groupby("movie_id").rating.agg(["size", "mean", "std"])
    s.columns = ["item_popularity", "item_avg_rating", "item_rating_std"]
    s["item_like_ratio"] = train.assign(l=train.rating >= 4).groupby("movie_id").l.mean()
    s["item_days_since_last_interaction"] = (cutoff - train.groupby("movie_id").timestamp.max()).dt.days
    f = items[["movie_id", "year", "primary_genre", "genre_list"]].merge(
        s, left_on="movie_id", right_index=True, how="left")
    f["item_popularity"] = f.item_popularity.fillna(0).astype(int)
    f["item_popularity_percentile"] = f.item_popularity.rank(pct=True)
    n = f.item_popularity
    f["item_bayesian_avg"] = (n * f.item_avg_rating.fillna(gmean) + prior_m * gmean) / (n + prior_m)
    f["item_rating_std"] = f.item_rating_std.fillna(0.0)
    f = f.rename(columns={"year": "item_release_year", "primary_genre": "item_primary_genre"})
    f["item_age_years"] = (cutoff.year - f.item_release_year).astype("Int64")
    for g in GENRES:
        f[genre_col(g)] = f.genre_list.map(lambda gl, g=g: int(g in gl)).astype("int8")
    return f.drop(columns=["genre_list"])


def build_user_features(train, users, items, item_feats, cutoff):
    gmean = train.rating.mean()
    g = train.groupby("user_id")
    f = pd.DataFrame({
        "user_activity_count": g.size(),
        "user_avg_rating": g.rating.mean(),
        "user_rating_std": g.rating.std().fillna(0.0),
        "user_like_ratio": train.assign(l=train.rating >= 4).groupby("user_id").l.mean(),
        "user_days_since_last_interaction": (cutoff - g.timestamp.max()).dt.days,
        "user_tenure_days": (g.timestamp.max() - g.timestamp.min()).dt.days + 1,
    })
    f["user_rating_bias"] = f.user_avg_rating - gmean
    ex = train[["user_id", "movie_id"]].merge(items[["movie_id", "genre_list"]], on="movie_id") \
        .explode("genre_list")
    f["user_n_distinct_genres"] = ex.groupby("user_id").genre_list.nunique()
    f["user_preferred_genre"] = ex.groupby("user_id").genre_list.agg(lambda s: s.value_counts().index[0])
    pop = train[["user_id", "movie_id"]].merge(item_feats[["movie_id", "item_popularity_percentile"]],
                                               on="movie_id")
    f["user_mainstreamness"] = pop.groupby("user_id").item_popularity_percentile.mean()
    demo = users.set_index("user_id")[["gender", "age", "occupation"]].rename(
        columns={"gender": "user_gender", "age": "user_age_bucket", "occupation": "user_occupation"})
    return f.join(demo, how="left").reset_index()


def build_user_genre_features(train, items):
    ex = train[["user_id", "movie_id", "rating"]].merge(items[["movie_id", "genre_list"]], on="movie_id") \
        .explode("genre_list").rename(columns={"genre_list": "genre"})
    ug = ex.groupby(["user_id", "genre"]).rating.agg(["size", "mean"]).reset_index()
    totals = train.groupby("user_id").size()
    ug["user_genre_share"] = ug["size"] / ug.user_id.map(totals)
    ug = ug.rename(columns={"mean": "user_genre_avg_rating"}).drop(columns="size")
    ug["user_genre_affinity"] = ug.user_genre_share * ug.user_genre_avg_rating / 5.0
    ug = ug[ug.genre.isin(GENRES)]
    return ug


class FeatureStore:
    """Read-only access layer over the offline feature store.

    The same object is used by training code and by the serving script, which
    guarantees identical feature definitions in both places.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.user = read_parquet(path(cfg, "features", "user_features.parquet")).set_index("user_id")
        self.item = read_parquet(path(cfg, "features", "item_features.parquet")).set_index("movie_id")
        self.user_genre = read_parquet(path(cfg, "features", "user_genre_features.parquet"))
        self.metadata = read_json(path(cfg, "features", "feature_registry.json"))

    def get_user_features(self, user_ids, columns=None) -> pd.DataFrame:
        df = self.user.reindex(user_ids)
        return df[columns] if columns else df

    def get_item_features(self, movie_ids, columns=None) -> pd.DataFrame:
        df = self.item.reindex(movie_ids)
        return df[columns] if columns else df

    def user_genre_matrix(self, user_ids) -> np.ndarray:
        """Dense (n_users x 18) matrix of user_genre_affinity aligned to GENRES."""
        piv = self.user_genre.pivot(index="user_id", columns="genre", values="user_genre_affinity")
        return piv.reindex(index=user_ids, columns=GENRES).fillna(0.0).to_numpy(np.float32)

    def item_genre_matrix(self, movie_ids) -> np.ndarray:
        return self.item.reindex(movie_ids)[GENRE_COLS].fillna(0).to_numpy(np.float32)

    def get_user_item_features(self, user_ids, movie_ids) -> pd.DataFrame:
        """On-demand user x item features for (user, candidate) pairs."""
        u = np.asarray(user_ids); i = np.asarray(movie_ids)
        ug = self.user_genre_matrix(np.unique(u))
        upos = pd.Series(np.arange(len(np.unique(u))), index=np.unique(u))[u].to_numpy()
        ig = self.item_genre_matrix(i)
        aff = (ug[upos] * ig).sum(1) / np.maximum(ig.sum(1), 1)
        gap = self.item.reindex(i).item_popularity_percentile.to_numpy() - \
            self.user.reindex(u).user_mainstreamness.to_numpy()
        return pd.DataFrame({"user_id": u, "movie_id": i, "ui_genre_affinity": aff, "ui_popularity_gap": gap})


def _write_catalog(cfg: dict, stats: dict) -> None:
    cat = pd.DataFrame(as_records())[["name", "entity", "dtype", "description", "derivation", "used_by"]]
    md = (f"# Feature Catalog\n\nFeature version **{FEATURE_VERSION}**, computed as of the train/test cutoff "
          f"`{stats['as_of']}` from training data only.\n\n"
          f"| Feature group | Entity key | Rows | File |\n|---|---|---|---|\n"
          f"| User features | user_id | {stats['user_rows']} | data/features/user_features.parquet |\n"
          f"| Item features | movie_id | {stats['item_rows']} | data/features/item_features.parquet |\n"
          f"| User-genre features | user_id, genre | {stats['user_genre_rows']} | data/features/user_genre_features.parquet |\n"
          f"| User-item features | user_id, movie_id | on demand | FeatureStore.get_user_item_features() |\n\n"
          f"## Feature definitions ({len(FEATURES)} entries)\n\n{df_to_markdown(cat)}\n\n"
          "## Reuse for training and prediction\n\n"
          "All models and the serving script read features through `src.features.build_features.FeatureStore`, "
          "never by recomputing them. Training reads them in `src/training/models.py` "
          "(Popularity -> `item_popularity`, Hybrid -> `user_genre_affinity`, `genre_*`, "
          "`item_popularity`). Prediction reads them in `src/serving/recommend.py`.\n")
    path(cfg, "reports", "feature_catalog.md").write_text(md, encoding="utf-8")


def run(cfg: dict) -> dict:
    with timed(log, "feature engineering"):
        train = read_parquet(path(cfg, "processed", "train.parquet"))
        users = read_parquet(path(cfg, "processed", "users.parquet"))
        items = read_parquet(path(cfg, "processed", "items.parquet"))
        cutoff = pd.Timestamp(read_json(path(cfg, "reports", "split_info.json"))["cutoff_utc"])

        item_f = build_item_features(train, items, cutoff, cfg["features"]["bayesian_prior_count"])
        user_f = build_user_features(train, users, items, item_f, cutoff)
        ug_f = build_user_genre_features(train, items)
        for name, df in [("user_features", user_f), ("item_features", item_f), ("user_genre_features", ug_f)]:
            df = df.assign(feature_version=FEATURE_VERSION, as_of=cutoff)
            write_parquet(df, path(cfg, "features", f"{name}.parquet"))
            log.info("%-20s %7d rows x %2d cols", name, len(df), df.shape[1])

        stats = {"as_of": cutoff.isoformat(), "user_rows": len(user_f), "item_rows": len(item_f),
                 "user_genre_rows": len(ug_f)}
        write_json({"version": FEATURE_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    **stats, "features": as_records()}, path(cfg, "features", "feature_registry.json"))
        _write_catalog(cfg, stats)
    return stats


if __name__ == "__main__":
    from src.utils import load_config
    run(load_config())

"""Prediction / serving entry point.

Loads a trained model plus the SAME feature store used in training and
returns top-K recommendations for a user, with feature-based explanations.

    python -m src.serving.recommend --user-id 1 --model hybrid_cf_content --k 10
"""
from __future__ import annotations

import argparse
import pickle

import pandas as pd

from src.features.build_features import FeatureStore
from src.utils import load_config, path, read_json, read_parquet


def recommend(cfg: dict, user_id: int, model_name: str | None = None, k: int = 10) -> pd.DataFrame:
    reg = read_json(path(cfg, "models", "model_registry.json"))
    model_name = model_name or "hybrid_cf_content"
    with open(path(cfg, "models", reg["models"][model_name]["path"]), "rb") as fh:
        model = pickle.load(fh)
    fs = FeatureStore(cfg)
    items = read_parquet(path(cfg, "processed", "items.parquet")).set_index("movie_id")

    known = set(model.user_index.user_id)
    if user_id not in known:                       # cold-start fallback -> popularity
        with open(path(cfg, "models", reg["models"]["popularity"]["path"]), "rb") as fh:
            model = pickle.load(fh)
        recs = fs.item.sort_values("item_popularity", ascending=False).index[:k].tolist()
    else:
        recs = model.recommend([user_id], k=k)[user_id]

    out = items.loc[recs, ["title", "genres"]].reset_index()
    ui = fs.get_user_item_features([user_id] * len(recs), recs)
    itf = fs.get_item_features(recs, ["item_popularity", "item_bayesian_avg"]).reset_index(drop=True)
    out = pd.concat([out, itf, ui[["ui_genre_affinity"]]], axis=1)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--model", default="hybrid_cf_content")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config()
    fs = FeatureStore(cfg)
    if args.user_id in fs.user.index:
        u = fs.get_user_features([args.user_id]).iloc[0]
        print(f"User {args.user_id}: {int(u.user_activity_count)} ratings, avg {u.user_avg_rating:.2f}, "
              f"preferred genre = {u.user_preferred_genre}")
    else:
        print(f"User {args.user_id} unseen in training -> cold-start popularity fallback")
    with pd.option_context("display.width", 160, "display.max_colwidth", 45):
        print(recommend(cfg, args.user_id, args.model, args.k).to_string(index=False))


if __name__ == "__main__":
    main()

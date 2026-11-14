"""Hyper-parameter tuning on an internal *validation* split .
The training data is split again by time: the last ``validation_fraction`` of
training interactions become a validation set. The hybrid model is fitted on
the earlier part, every weight combination in the grid is scored by NDCG@K on
the validation users, and the best combination is returned. The final model is
then refitted on the full training data with the selected weights.

Feature values used during tuning are recomputed from the *fit* part only
(``ValidationFeatureView``) using the same formulas as the feature store, so no
validation information leaks into tuning.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation import metrics as M
from src.training.models import HybridRecommender, TrainingContext
from src.utils import get_logger

log = get_logger("tuning")


class ValidationFeatureView:
    """Feature-store look-alike whose behavioural features come from the fit split only."""

    def __init__(self, store, R_fit: sp.csr_matrix, item_ids: np.ndarray):
        self.store, self.item_ids = store, item_ids
        self.R = R_fit
        self.pop = np.asarray((R_fit > 0).sum(0)).ravel()
        self.metadata = store.metadata

    def item_genre_matrix(self, movie_ids):                 # static content
        return self.store.item_genre_matrix(movie_ids)

    def user_genre_matrix(self, user_ids):
        # user_genre_affinity = share * avg / 5 = sum(ratings in genre) / (5 * n_user_ratings)
        ig = self.store.item_genre_matrix(self.item_ids)
        n_u = np.asarray((self.R > 0).sum(1)).ravel()
        return np.asarray(self.R @ ig) / (5 * np.maximum(n_u, 1))[:, None]

    def get_item_features(self, movie_ids, columns=None):
        return pd.DataFrame({"item_popularity": self.pop}, index=self.item_ids).reindex(movie_ids)


def tune_hybrid(cfg: dict, ctx: TrainingContext, train: pd.DataFrame, base_params: dict):
    tcfg = cfg["tuning"]
    k = cfg["evaluation"]["k"]
    cut = train.timestamp.quantile(1 - tcfg["validation_fraction"])
    fit, val = train[train.timestamp < cut], train[train.timestamp >= cut]
    row = pd.Series(ctx.user_index.row.to_numpy(), index=ctx.user_index.user_id.to_numpy())
    col = pd.Series(ctx.item_index.col.to_numpy(), index=ctx.item_index.movie_id.to_numpy())
    R_fit = sp.csr_matrix((fit.rating.astype(np.float32).to_numpy(),
                           (row[fit.user_id].to_numpy(), col[fit.movie_id].to_numpy())),
                          shape=ctx.matrix.shape)
    fit_counts = fit.user_id.value_counts()
    warm = fit_counts[fit_counts >= cfg["transformation"]["min_train_interactions_per_user"]].index
    val = val[val.user_id.isin(warm) & (val.rating >= cfg["transformation"]["relevance_threshold"])]
    rel = val.groupby("user_id").movie_id.apply(set)

    vctx = TrainingContext(matrix=R_fit, user_index=ctx.user_index, item_index=ctx.item_index,
                           features=ValidationFeatureView(ctx.features, R_fit, ctx.item_ids), seed=ctx.seed)
    model = HybridRecommender(**base_params).fit(vctx)
    rows = row[rel.index].to_numpy()
    cf, content, pop = model.components(rows)
    seen = model.seen[rows].toarray() > 0

    results = []
    for a, b in itertools.product(tcfg["alpha_grid"], tcfg["beta_grid"]):
        g = round(1 - a - b, 3)
        if g < 0:
            continue
        s = a * cf + b * content + g * pop
        s[seen] = -np.inf
        top = np.argpartition(-s, k, 1)[:, :k]
        order = np.take_along_axis(s, top, 1).argsort(1)[:, ::-1]
        top = np.take_along_axis(top, order, 1)
        recs = ctx.item_ids[top]
        ndcg = float(np.mean([M.ndcg_at_k(r.tolist(), t, k) for r, t in zip(recs, rel.values)]))
        cov = len(set(recs.ravel())) / len(ctx.item_ids)
        results.append({"alpha_cf": a, "beta_content": b, "gamma_popularity": g,
                        f"val_ndcg_at_{k}": ndcg, f"val_coverage_at_{k}": cov})
    res = pd.DataFrame(results).sort_values(f"val_ndcg_at_{k}", ascending=False)
    best = res.iloc[0]
    log.info("validation users=%d  grid=%d  best a=%.2f b=%.2f g=%.2f NDCG@%d=%.4f", len(rel), len(res),
             best.alpha_cf, best.beta_content, best.gamma_popularity, k, best[f"val_ndcg_at_{k}"])
    return {"alpha_cf": float(best.alpha_cf), "beta_content": float(best.beta_content),
            "gamma_popularity": float(best.gamma_popularity)}, res

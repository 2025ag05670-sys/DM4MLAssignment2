"""Recommender models.

All models share one interface:

    fit(ctx)                         -> self
    score(user_rows) -> np.ndarray   (n_users x n_items, higher = better)
    recommend(user_ids, k)           -> {user_id: [movie_id, ...]}  (already-seen items excluded)

Models
------
* PopularityRecommender      - non-personalised baseline (most-rated items).
* ItemKNNRecommender         - item-based collaborative filtering (cosine with shrinkage).
* MatrixFactorizationALS     - explicit-rating matrix factorisation with user/item
                               biases trained by Alternating Least Squares; also
                               predicts ratings (RMSE / MAE).
* HybridRecommender          - weighted blend of item-kNN CF, content-based genre
                               affinity (from the feature store) and a popularity prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.build_features import FeatureStore


@dataclass
class TrainingContext:
    matrix: sp.csr_matrix          # users x items ratings (train)
    user_index: pd.DataFrame       # user_id -> row
    item_index: pd.DataFrame       # movie_id -> col
    features: FeatureStore
    seed: int = 42

    @property
    def item_ids(self) -> np.ndarray:
        return self.item_index.movie_id.to_numpy()

    def rows(self, user_ids) -> np.ndarray:
        m = pd.Series(self.user_index.row.to_numpy(), index=self.user_index.user_id.to_numpy())
        return m.reindex(user_ids).to_numpy()


class BaseRecommender:
    name = "base"

    def __init__(self, **params):
        self.params = params

    def fit(self, ctx: TrainingContext):
        self.item_ids = ctx.item_ids
        self.user_index = ctx.user_index
        self.seen = ctx.matrix.copy().tocsr()
        self.seen.data[:] = 1
        self._fit(ctx)
        return self

    def _fit(self, ctx):  # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, user_rows: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _rows(self, user_ids):
        m = pd.Series(self.user_index.row.to_numpy(), index=self.user_index.user_id.to_numpy())
        return m.reindex(user_ids).to_numpy()

    def recommend(self, user_ids, k: int = 10, batch: int = 1024, exclude_seen: bool = True) -> dict:
        user_ids = np.asarray(user_ids)
        rows = self._rows(user_ids)
        out = {}
        for s in range(0, len(user_ids), batch):
            r = rows[s:s + batch]
            sc = self.score(r).astype(np.float64)
            if exclude_seen:
                sc[self.seen[r].toarray() > 0] = -np.inf
            top = np.argpartition(-sc, k, axis=1)[:, :k]
            order = np.take_along_axis(sc, top, 1).argsort(1)[:, ::-1]
            top = np.take_along_axis(top, order, 1)
            for uid, t in zip(user_ids[s:s + batch], top):
                out[int(uid)] = self.item_ids[t].tolist()
        return out


def _minmax_rows(x: np.ndarray) -> np.ndarray:
    lo = x.min(1, keepdims=True); hi = x.max(1, keepdims=True)
    return (x - lo) / np.where(hi - lo > 0, hi - lo, 1)


# ----------------------------------------------------------- Popularity ----
class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def _fit(self, ctx):
        f = ctx.features.get_item_features(self.item_ids)
        col = {"interaction_count": "item_popularity", "bayesian_avg": "item_bayesian_avg"}[
            self.params.get("score", "interaction_count")]
        self.item_scores = f[col].fillna(0).to_numpy(np.float32)
        self.rating_pred = f["item_bayesian_avg"].to_numpy(np.float32)   # for RMSE baseline

    def score(self, user_rows):
        return np.tile(self.item_scores, (len(user_rows), 1))

    def predict(self, user_ids, movie_ids):
        pos = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids).reindex(movie_ids).to_numpy()
        return self.rating_pred[pos]


# ------------------------------------------------------------- Item-kNN ----
class ItemKNNRecommender(BaseRecommender):
    name = "item_knn_cf"

    def _fit(self, ctx):
        x = self.seen.astype(np.float32)
        co = (x.T @ x).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.params.get("shrinkage", 10.0))
        np.fill_diagonal(sim, 0.0)
        n = self.params.get("neighbours", 100)
        if n and n < sim.shape[0]:
            thresh = -np.partition(-sim, n, axis=0)[n - 1]
            sim[sim < thresh] = 0.0
        self.sim = sp.csr_matrix(sim.astype(np.float32))

    def score(self, user_rows):
        return (self.seen[user_rows] @ self.sim).toarray()


# --------------------------------------------------- Matrix factorisation ----
class MatrixFactorizationALS(BaseRecommender):
    name = "matrix_factorization_als"

    def _fit(self, ctx):
        rng = np.random.default_rng(ctx.seed)
        R = ctx.matrix.tocsr().astype(np.float64)
        Rc = R.tocsc()
        k = self.params.get("n_factors", 50)
        lam = self.params.get("reg_factors", 0.08)
        lb = self.params.get("reg_bias", 5.0)
        n_u, n_i = R.shape
        self.mu = R.data.mean()

        # regularised biases (a few alternating passes)
        bu, bi = np.zeros(n_u), np.zeros(n_i)
        coo = R.tocoo()
        for _ in range(5):
            res = coo.data - self.mu - bu[coo.row]
            bi = np.bincount(coo.col, res, n_i) / (np.bincount(coo.col, minlength=n_i) + lb)
            res = coo.data - self.mu - bi[coo.col]
            bu = np.bincount(coo.row, res, n_u) / (np.bincount(coo.row, minlength=n_u) + lb)
        self.bu, self.bi = bu, bi

        # residuals for factor model
        resid = coo.data - self.mu - bu[coo.row] - bi[coo.col]
        E = sp.csr_matrix((resid, (coo.row, coo.col)), shape=R.shape)
        Ec = E.tocsc()
        P = rng.normal(0, 0.1, (n_u, k))
        Q = rng.normal(0, 0.1, (n_i, k))
        I = np.eye(k)
        self.loss_history = []
        for _ in range(self.params.get("n_iterations", 15)):
            for u in range(n_u):
                s, e = E.indptr[u], E.indptr[u + 1]
                if s == e:
                    continue
                Qi = Q[E.indices[s:e]]
                P[u] = np.linalg.solve(Qi.T @ Qi + lam * (e - s) * I, Qi.T @ E.data[s:e])
            for i in range(n_i):
                s, e = Ec.indptr[i], Ec.indptr[i + 1]
                if s == e:
                    continue
                Pu = P[Ec.indices[s:e]]
                Q[i] = np.linalg.solve(Pu.T @ Pu + lam * (e - s) * I, Pu.T @ Ec.data[s:e])
            pred = (P[coo.row] * Q[coo.col]).sum(1)
            self.loss_history.append(float(np.sqrt(np.mean((resid - pred) ** 2))))
        self.P, self.Q = P.astype(np.float32), Q.astype(np.float32)
        del Rc

    def score(self, user_rows):
        return self.mu + self.bu[user_rows, None] + self.bi[None, :] + self.P[user_rows] @ self.Q.T

    def predict(self, user_ids, movie_ids):
        r = self._rows(user_ids)
        c = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids).reindex(movie_ids).to_numpy()
        p = self.mu + self.bu[r] + self.bi[c] + (self.P[r] * self.Q[c]).sum(1)
        return np.clip(p, 1, 5)


# ------------------------------------------------------------------ Hybrid ----
class HybridRecommender(BaseRecommender):
    name = "hybrid_cf_content"

    def _fit(self, ctx):
        knn_params = {k: v for k, v in self.params.items() if k in ("shrinkage", "neighbours")}
        self.knn = ItemKNNRecommender(**knn_params)
        self.knn.fit(ctx)
        users = ctx.user_index.user_id.to_numpy()
        self.user_genre = ctx.features.user_genre_matrix(users)          # feature store
        ig = ctx.features.item_genre_matrix(self.item_ids)               # feature store
        self.item_genre = ig / np.maximum(ig.sum(1, keepdims=True), 1)
        pop = ctx.features.get_item_features(self.item_ids)["item_popularity"].fillna(0) \
            .to_numpy(np.float32)
        self.pop = pop / max(pop.max(), 1)

    def components(self, user_rows):
        """Return the three normalised score components (used for weight tuning)."""
        cf = _minmax_rows(self.knn.score(user_rows))
        content = _minmax_rows(self.user_genre[user_rows] @ self.item_genre.T)
        return cf, content, np.broadcast_to(self.pop[None, :], cf.shape)

    def score(self, user_rows):
        a = self.params.get("alpha_cf", 0.6)
        b = self.params.get("beta_content", 0.1)
        g = self.params.get("gamma_popularity", 0.3)
        cf, content, pop = self.components(user_rows)
        return a * cf + b * content + g * pop


MODEL_CLASSES = {
    "popularity": PopularityRecommender,
    "item_knn": ItemKNNRecommender,
    "matrix_factorization": MatrixFactorizationALS,
    "hybrid": HybridRecommender,
}

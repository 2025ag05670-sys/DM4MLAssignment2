"""Ranking, beyond-accuracy and rating-prediction metrics (pure numpy, unit-tested)."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def precision_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(recs[:k]) if it in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def hit_rate_at_k(recs: list, relevant: set, k: int) -> float:
    return float(bool(set(recs[:k]) & relevant))


def catalog_coverage(all_recs: dict, catalog_size: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list."""
    return len({i for r in all_recs.values() for i in r}) / catalog_size


def intra_list_diversity(recs: list, item_vectors: dict) -> float:
    """1 - mean pairwise cosine similarity of item genre vectors inside one list."""
    vecs = [item_vectors[i] for i in recs if i in item_vectors]
    if len(vecs) < 2:
        return 0.0
    sims = []
    for a, b in combinations(vecs, 2):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        sims.append(float(a @ b / (na * nb)) if na and nb else 0.0)
    return 1.0 - float(np.mean(sims))


def novelty(recs: list, item_pop_prob: dict) -> float:
    """Mean self-information -log2 p(i) of recommended items (higher = less mainstream)."""
    vals = [-np.log2(item_pop_prob[i]) for i in recs if item_pop_prob.get(i, 0) > 0]
    return float(np.mean(vals)) if vals else 0.0


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

"""Stage 7 - Offline evaluation and model comparison.

Protocol
--------
* Test users = warm users that have at least one *relevant* test item
  (rating >= ``relevance_threshold``) after the time cut-off.
* Each model ranks all training-catalogue items the user has not yet seen
  and returns the top-K.
* Ranking metrics: Precision@K, Recall@K, NDCG@K, HitRate@K (K = 5 and 10).
* Beyond-accuracy: catalogue Coverage@K, intra-list Diversity@K (genre
  cosine), Novelty@K.
* Rating metrics (RMSE, MAE) for models that predict explicit ratings, on
  every warm test rating.

Writes ``reports/model_comparison.{csv,md}``, ``reports/figures/*.png``,
per-model sample recommendations, and logs everything to the MLflow runs
created by the training stage.
"""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation import metrics as M  # noqa: E402
from src.features.build_features import GENRE_COLS  # noqa: E402
from src.training.train import setup_mlflow  # noqa: E402
from src.utils import (df_to_markdown, get_logger, path, read_json, read_parquet, timed,  # noqa: E402
                       write_json)

log = get_logger("evaluation")


def evaluate_model(model, test: pd.DataFrame, k_list, item_vecs, pop_prob, catalog_size):
    rel = test[test.relevant].groupby("user_id").movie_id.apply(set)
    users = rel.index.to_numpy()
    kmax = max(k_list)
    recs = model.recommend(users, k=kmax)
    out = {}
    for k in k_list:
        p, r, n, h, d, nov = [], [], [], [], [], []
        for u in users:
            lst, rs = recs[int(u)], rel[u]
            p.append(M.precision_at_k(lst, rs, k)); r.append(M.recall_at_k(lst, rs, k))
            n.append(M.ndcg_at_k(lst, rs, k)); h.append(M.hit_rate_at_k(lst, rs, k))
            d.append(M.intra_list_diversity(lst[:k], item_vecs)); nov.append(M.novelty(lst[:k], pop_prob))
        out.update({f"precision_at_{k}": np.mean(p), f"recall_at_{k}": np.mean(r),
                    f"ndcg_at_{k}": np.mean(n), f"hit_rate_at_{k}": np.mean(h),
                    f"coverage_at_{k}": M.catalog_coverage({u: v[:k] for u, v in recs.items()}, catalog_size),
                    f"diversity_at_{k}": np.mean(d), f"novelty_at_{k}": np.mean(nov)})
    if hasattr(model, "predict"):
        pred = model.predict(test.user_id.to_numpy(), test.movie_id.to_numpy())
        out["rmse"] = M.rmse(test.rating, pred)
        out["mae"] = M.mae(test.rating, pred)
    out["n_eval_users"] = len(users)
    return out, recs


def _plot(df: pd.DataFrame, k: int, fp):
    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"coverage_at_{k}", f"diversity_at_{k}"]
    labels = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", f"Coverage@{k}", f"Diversity@{k}"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(df)
    x = np.arange(len(cols))
    palette = ["#9aa5b1", "#3b82f6", "#f59e0b", "#10b981"]
    for j, (_, row) in enumerate(df.iterrows()):
        vals = [row[c] for c in cols]
        bars = ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=row["model"], color=palette[j % 4])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("score")
    ax.set_title(f"Model comparison on the time-based test set (K={k})")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fp, dpi=150)
    plt.close(fig)


def run(cfg: dict) -> pd.DataFrame:
    ecfg = cfg["evaluation"]
    setup_mlflow(cfg)
    with timed(log, "evaluation"):
        test = read_parquet(path(cfg, "processed", "test.parquet"))
        reg = read_json(path(cfg, "models", "model_registry.json"))
        items = read_parquet(path(cfg, "features", "item_features.parquet"))
        titles = read_parquet(path(cfg, "processed", "items.parquet")).set_index("movie_id").title
        item_vecs = dict(zip(items.movie_id, items[GENRE_COLS].to_numpy(float)))
        train = read_parquet(path(cfg, "processed", "train.parquet"))
        pop_prob = (train.movie_id.value_counts() / train.user_id.nunique()).to_dict()
        catalog_size = train.movie_id.nunique()

        rows, samples = [], {}
        for name, meta in reg["models"].items():
            with open(path(cfg, "models", meta["path"]), "rb") as fh:
                model = pickle.load(fh)
            res, recs = evaluate_model(model, test, ecfg["k_list"], item_vecs, pop_prob, catalog_size)
            rows.append({"model": name, **res, "train_time_sec": meta["train_time_sec"]})
            some = list(recs)[:5]
            samples[name] = {u: [titles.get(i, str(i)) for i in recs[u][:ecfg["k"]]] for u in some}
            with mlflow.start_run(run_id=meta["run_id"]):
                mlflow.log_metrics({k: float(v) for k, v in res.items()})
            log.info("%-26s P@10=%.4f R@10=%.4f NDCG@10=%.4f Cov@10=%.3f Div@10=%.3f %s", name,
                     res["precision_at_10"], res["recall_at_10"], res["ndcg_at_10"], res["coverage_at_10"],
                     res["diversity_at_10"], f"RMSE={res['rmse']:.4f}" if "rmse" in res else "")

        df = pd.DataFrame(rows)
        df.to_csv(path(cfg, "reports", "model_comparison.csv"), index=False)
        write_json(samples, path(cfg, "reports", "sample_recommendations.json"))
        k = ecfg["k"]
        main_cols = ["model", f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"hit_rate_at_{k}",
                     f"coverage_at_{k}", f"diversity_at_{k}", f"novelty_at_{k}", "rmse", "mae"]
        k5 = ["model", "precision_at_5", "recall_at_5", "ndcg_at_5", "coverage_at_5", "diversity_at_5"]
        show = df[main_cols].copy()
        for c in ["rmse", "mae"]:
            show[c] = show[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.4f}")
        best = df.loc[df[ecfg["primary_metric"]].idxmax(), "model"]
        split = read_json(path(cfg, "reports", "split_info.json"))
        md = (f"# Model Evaluation Results\n\nTest set: interactions after `{split['cutoff_utc']}` "
              f"(time-based split), {int(df.n_eval_users.iloc[0])} warm users with at least one relevant "
              f"item (rating >= {cfg['transformation']['relevance_threshold']}). Already-seen items are "
              f"excluded from every ranking. Catalogue size = {catalog_size} items.\n\n"
              f"## K = {k}\n\n{df_to_markdown(show)}\n\n## K = 5\n\n{df_to_markdown(df[k5])}\n\n"
              f"**Best model by {ecfg['primary_metric']}: `{best}`**\n\n"
              "RMSE/MAE are computed on every warm test rating; the popularity model predicts the item's "
              "Bayesian-average rating, which acts as the rating baseline for matrix factorisation. "
              "Ranking-only models (item-kNN, hybrid) do not predict ratings (n/a).\n\n"
              f"![comparison](figures/model_comparison_k{k}.png)\n")
        path(cfg, "reports", "model_comparison.md").write_text(md, encoding="utf-8")
        _plot(df, k, path(cfg, "reports", "figures", f"model_comparison_k{k}.png"))
        _plot(df, 5, path(cfg, "reports", "figures", "model_comparison_k5.png"))

        with mlflow.start_run(run_id=reg["parent_run_id"]):
            mlflow.log_artifact(str(path(cfg, "reports", "model_comparison.csv")), "evaluation")
            mlflow.log_artifact(str(path(cfg, "reports", "model_comparison.md")), "evaluation")
            mlflow.log_artifact(str(path(cfg, "reports", "figures", f"model_comparison_k{k}.png")), "evaluation")
            mlflow.log_artifact(str(path(cfg, "reports", "validation_report.md")), "data_quality")
            mlflow.set_tag("best_model", best)
            mlflow.log_metric(f"best_{ecfg['primary_metric']}", float(df[ecfg["primary_metric"]].max()))
        log.info("best model by %s: %s", ecfg["primary_metric"], best)
    return df


if __name__ == "__main__":
    from src.utils import load_config
    run(load_config())

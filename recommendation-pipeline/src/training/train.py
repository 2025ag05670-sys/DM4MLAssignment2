"""Stage 6 - Model training with MLflow experiment tracking.

One MLflow *parent* run per pipeline execution, with one *nested* run per
model. Each child run records the model's hyper-parameters, training time, the
data / feature versions it was trained on and the pickled model artifact. The
run IDs are written to ``models/model_registry.json`` so the evaluation stage
can attach metrics to the same runs.
"""
from __future__ import annotations

import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import scipy.sparse as sp

from src.features.build_features import FeatureStore
from src.training.models import MODEL_CLASSES, TrainingContext
from src.training.tuning import tune_hybrid
from src.utils import (PROJECT_ROOT, get_logger, md5, path, read_json, read_parquet, set_seed,
                       timed, write_json)

log = get_logger("training")


def setup_mlflow(cfg: dict) -> str:
    os.chdir(PROJECT_ROOT)  # relative sqlite / artifact paths resolve inside the project
    mcfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    exp = mlflow.get_experiment_by_name(mcfg["experiment_name"])
    if exp is None:
        art = (PROJECT_ROOT / mcfg["artifact_location"]).resolve().as_uri()
        exp_id = mlflow.create_experiment(mcfg["experiment_name"], artifact_location=art)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(mcfg["experiment_name"])
    return exp_id


def load_context(cfg: dict) -> TrainingContext:
    return TrainingContext(
        matrix=sp.load_npz(path(cfg, "processed", "train_matrix.npz")).tocsr(),
        user_index=read_parquet(path(cfg, "processed", "user_index.parquet")),
        item_index=read_parquet(path(cfg, "processed", "item_index.parquet")),
        features=FeatureStore(cfg), seed=cfg["project"]["seed"])


def run(cfg: dict) -> dict:
    set_seed(cfg["project"]["seed"])
    setup_mlflow(cfg)
    with timed(log, "training"):
        ctx = load_context(cfg)
        split = read_json(path(cfg, "reports", "split_info.json"))
        data_hash = md5(path(cfg, "processed", "train.parquet"))
        registry = {"trained_at_utc": datetime.now(timezone.utc).isoformat(), "models": {}}

        with mlflow.start_run(run_name=f"pipeline_{datetime.now():%Y%m%d_%H%M%S}") as parent:
            registry["parent_run_id"] = parent.info.run_id
            mlflow.set_tags({"stage": "pipeline", "dataset": cfg["ingestion"]["dataset"]})
            mlflow.log_params({"seed": cfg["project"]["seed"], "split_strategy": split["strategy"],
                               "split_cutoff": split["cutoff_utc"], "train_rows": split["train_rows"],
                               "test_rows": split["test_rows"], "train_data_md5": data_hash,
                               "feature_version": ctx.features.metadata["version"]})
            mlflow.log_artifact(cfg["_config_path"], artifact_path="config")

            for key, cls in MODEL_CLASSES.items():
                params = dict(cfg["models"].get(key, {}))
                if key == "hybrid":
                    params.update(cfg["models"]["item_knn"])
                with mlflow.start_run(run_name=cls.name, nested=True) as child:
                    if key == "hybrid" and cfg.get("tuning", {}).get("enabled"):
                        train_df = read_parquet(path(cfg, "processed", "train.parquet"))
                        best, grid = tune_hybrid(cfg, ctx, train_df, params)
                        params.update(best)
                        gp = path(cfg, "reports", "hybrid_tuning_grid.csv")
                        grid.to_csv(gp, index=False)
                        mlflow.log_artifact(str(gp), artifact_path="tuning")
                        mlflow.log_metric("val_best_ndcg", float(grid.iloc[0, 3]))
                        mlflow.set_tag("tuned_on", "validation split (last 10% of train)")
                    t0 = time.perf_counter()
                    model = cls(**params).fit(ctx)
                    train_s = time.perf_counter() - t0
                    mp = Path(path(cfg, "models", f"{cls.name}.pkl"))
                    with open(mp, "wb") as fh:
                        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    mlflow.set_tags({"model": cls.name, "model_family": key})
                    mlflow.log_params({"model": cls.name, **params,
                                       "train_data_md5": data_hash,
                                       "feature_version": ctx.features.metadata["version"]})
                    mlflow.log_metric("train_time_sec", train_s)
                    if hasattr(model, "loss_history"):
                        for step, l in enumerate(model.loss_history):
                            mlflow.log_metric("train_residual_rmse", l, step=step)
                    mlflow.log_artifact(str(mp), artifact_path="model")
                    registry["models"][cls.name] = {"run_id": child.info.run_id, "path": str(mp.name),
                                                    "train_time_sec": round(train_s, 2), "params": params}
                    log.info("trained %-26s in %6.1fs  run_id=%s", cls.name, train_s, child.info.run_id)

        write_json(registry, path(cfg, "models", "model_registry.json"))
    return registry


if __name__ == "__main__":
    from src.utils import load_config
    run(load_config())

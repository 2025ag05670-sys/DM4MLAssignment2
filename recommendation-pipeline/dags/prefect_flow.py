"""Prefect 3 orchestration of the recommendation pipeline.

Run once (local, no server needed):
    python dags/prefect_flow.py

Run with the Prefect UI (for the run screenshot):
    prefect server start                                  # terminal 1 -> http://127.0.0.1:4200
    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
    python dags/prefect_flow.py                           # terminal 2

Schedule daily at 02:00 (creates a deployment served by this process):
    python dags/prefect_flow.py --serve

Task graph (each task receives the upstream result, so Prefect tracks the
dependencies explicitly):

    ingest -> validate -> prepare -> transform -> build_features -> train -> evaluate
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from prefect import flow, get_run_logger, task  # noqa: E402

from src.evaluation import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.validation import validate  # noqa: E402


@task(name="ingest_data", retries=3, retry_delay_seconds=10)
def ingest_task(cfg: dict) -> dict:
    m = ingest.run(cfg)
    return {"rows": {t: o["rows"] for t, o in m["outputs"].items()}}


@task(name="validate_data")
def validate_task(cfg: dict, upstream: dict) -> dict:
    rep = validate.run(cfg)                      # raises -> flow fails on quality gate
    return {s["table"]: s["rejected_rows"] for s in rep["summary"]}


@task(name="prepare_data")
def prepare_task(cfg: dict, upstream: dict) -> dict:
    return prepare.run(cfg)["rows"]


@task(name="transform_data")
def transform_task(cfg: dict, upstream: dict) -> dict:
    info = transform.run(cfg)
    return {"train_rows": info["train_rows"], "test_rows": info["test_rows"]}


@task(name="build_features")
def features_task(cfg: dict, upstream: dict) -> dict:
    return build_features.run(cfg)


@task(name="train_models")
def train_task(cfg: dict, upstream: dict) -> dict:
    reg = train.run(cfg)
    return {k: v["run_id"] for k, v in reg["models"].items()}


@task(name="evaluate_models")
def evaluate_task(cfg: dict, upstream: dict) -> list:
    df = evaluate.run(cfg)
    return df[["model", "precision_at_10", "recall_at_10", "ndcg_at_10"]].round(4).to_dict("records")


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None):
    logger = get_run_logger()
    cfg = load_config(config_path)
    record = {"flow": "recommendation-pipeline", "started_utc": datetime.now(timezone.utc).isoformat(),
              "tasks": []}
    try:
        from prefect.runtime import flow_run
        record["flow_run_id"], record["flow_run_name"] = str(flow_run.id), str(flow_run.name)
    except Exception:  # noqa: BLE001 - metadata only
        pass
    t0 = time.perf_counter()

    def step(name, fn, *args):
        start = time.perf_counter() - t0
        result = fn(*args)                 # an exception here fails the flow and all downstream tasks
        record["tasks"].append({"task": name, "state": "Completed", "start_s": round(start, 2),
                                "end_s": round(time.perf_counter() - t0, 2)})
        return result

    r1 = step("ingest_data", ingest_task, cfg)
    r2 = step("validate_data", validate_task, cfg, r1)
    r3 = step("prepare_data", prepare_task, cfg, r2)
    r4 = step("transform_data", transform_task, cfg, r3)
    r5 = step("build_features", features_task, cfg, r4)
    r6 = step("train_models", train_task, cfg, r5)
    r7 = step("evaluate_models", evaluate_task, cfg, r6)
    for row in r7:
        logger.info("RESULT %s", row)
    record.update({"state": "Completed", "total_seconds": round(time.perf_counter() - t0, 2), "results": r7})
    Path(cfg["paths"]["reports"], "orchestration_run.json").write_text(json.dumps(record, indent=2, default=str))
    return r7


if __name__ == "__main__":
    if "--serve" in sys.argv:
        recommendation_pipeline.serve(name="daily-recommendation-pipeline", cron="0 2 * * *")
    else:
        recommendation_pipeline()

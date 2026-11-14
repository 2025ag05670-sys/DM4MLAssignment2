"""Plain-Python orchestrator (no scheduler needed).

    python run_pipeline.py                      # run all stages in order
    python run_pipeline.py --stage validation   # run one stage
    python run_pipeline.py --from-stage training

The same stage functions are wrapped as tasks in ``dags/prefect_flow.py``
(Prefect) and ``dags/airflow_dag.py`` (Apache Airflow).
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.evaluation import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.utils import get_logger, load_config, path, write_json  # noqa: E402
from src.validation import validate  # noqa: E402

# Ordered DAG: each stage depends on the one before it.
STAGES = [
    ("ingestion", ingest.run),
    ("validation", validate.run),
    ("preparation", prepare.run),
    ("transformation", transform.run),
    ("features", build_features.run),
    ("training", train.run),
    ("evaluation", evaluate.run),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", choices=[s for s, _ in STAGES])
    ap.add_argument("--from-stage", choices=[s for s, _ in STAGES])
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("orchestrator")

    names = [s for s, _ in STAGES]
    if args.stage:
        selected = [args.stage]
    elif args.from_stage:
        selected = names[names.index(args.from_stage):]
    else:
        selected = names

    status, t0 = {}, time.perf_counter()
    for name, fn in STAGES:
        if name not in selected:
            continue
        s = time.perf_counter()
        try:
            fn(cfg)
            status[name] = {"state": "SUCCESS", "seconds": round(time.perf_counter() - s, 1)}
        except Exception as err:
            status[name] = {"state": "FAILED", "error": repr(err)}
            log.exception("stage %s failed - downstream stages skipped", name)
            write_json(status, path(cfg, "reports", "last_run_status.json"))
            raise SystemExit(1)
    log.info("PIPELINE SUCCESS in %.1fs: %s", time.perf_counter() - t0,
             {k: v["seconds"] for k, v in status.items()})
    write_json(status, path(cfg, "reports", "last_run_status.json"))


if __name__ == "__main__":
    main()

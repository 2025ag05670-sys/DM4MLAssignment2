"""Apache Airflow DAG (alternative to the Prefect flow - same stage functions).

Copy or symlink this file into ``$AIRFLOW_HOME/dags`` and set the
``RECSYS_PROJECT_ROOT`` environment variable to the repository path, e.g.

    export RECSYS_PROJECT_ROOT=$(pwd)
    airflow standalone            # UI at http://localhost:8080

Schedule: daily at 02:00, no catch-up, 2 retries per task.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = os.environ.get("RECSYS_PROJECT_ROOT", os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _stage(module_path: str):
    def _run(**_):
        import importlib

        from src.utils import load_config
        importlib.import_module(module_path).run(load_config())
    return _run


default_args = {"owner": "ml-team", "retries": 2, "retry_delay": timedelta(minutes=2)}

with DAG(
    dag_id="recommendation_pipeline",
    description="MovieLens end-to-end data management + recommender pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args=default_args,
    tags=["recsys", "dm4ml"],
) as dag:
    ingest = PythonOperator(task_id="ingest_data", python_callable=_stage("src.ingestion.ingest"))
    validate = PythonOperator(task_id="validate_data", python_callable=_stage("src.validation.validate"))
    prepare = PythonOperator(task_id="prepare_data", python_callable=_stage("src.preparation.prepare"))
    transform = PythonOperator(task_id="transform_data", python_callable=_stage("src.transformation.transform"))
    features = PythonOperator(task_id="build_features", python_callable=_stage("src.features.build_features"))
    train = PythonOperator(task_id="train_models", python_callable=_stage("src.training.train"))
    evaluate = PythonOperator(task_id="evaluate_models", python_callable=_stage("src.evaluation.evaluate"))

    ingest >> validate >> prepare >> transform >> features >> train >> evaluate

"""Unit tests for metrics and validation rules.  Run:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import metrics as M  # noqa: E402
from src.validation.validate import validate_interactions, validate_users  # noqa: E402

V = {"rating_min": 1, "rating_max": 5, "rating_integer_only": True, "timestamp_min": "1995-01-01",
     "timestamp_max": "2026-01-01", "valid_genders": ["M", "F"], "valid_ages": [1, 18, 25, 35, 45, 50, 56],
     "occupation_range": [0, 20]}


# ---------------------------------------------------------------- metrics ----
def test_precision_recall():
    recs, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert M.precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)
    assert M.recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)


def test_ndcg_perfect_and_zero():
    assert M.ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert M.ndcg_at_k([4, 5, 6], {1, 2, 3}, 3) == 0.0


def test_ndcg_position_sensitivity():
    assert M.ndcg_at_k([1, 9, 9], {1}, 3) > M.ndcg_at_k([9, 9, 1], {1}, 3)


def test_coverage_and_diversity():
    assert M.catalog_coverage({1: [1, 2], 2: [2, 3]}, 10) == pytest.approx(0.3)
    v = {1: np.array([1, 0]), 2: np.array([1, 0]), 3: np.array([0, 1])}
    assert M.intra_list_diversity([1, 2], v) == pytest.approx(0.0)
    assert M.intra_list_diversity([1, 3], v) == pytest.approx(1.0)


def test_rmse_mae():
    assert M.rmse([1, 3], [2, 5]) == pytest.approx(np.sqrt(2.5))
    assert M.mae([1, 3], [2, 5]) == pytest.approx(1.5)


# ------------------------------------------------------------- validation ----
def _rules(engine):
    return {(r["rule"], r["column"]): r["failed_rows"] for r in engine.results}


def test_interaction_rules_catch_each_problem():
    df = pd.DataFrame({
        "user_id":   ["1", "1", "",  "1",    "1", "1",          "77", "1",  "1"],
        "movie_id":  ["10", "10", "10", "11", "12", "13",       "10", "999", "14"],
        "rating":    ["5", "5", "4", "five", "9", "3",          "4", "4",   "3"],
        "timestamp": ["978300760", "978300760", "978300760", "978300760", "978300760",
                      "not_a_date", "978300760", "978300760", "978300760"],
    })
    eng = validate_interactions(df, V, valid_users={1}, valid_items={10, 11, 12, 13, 14})
    r = _rules(eng)
    assert r[("exact_duplicate", "*")] == 1
    assert r[("missing_value", "user_id")] == 1
    assert r[("invalid_type", "rating")] == 1
    assert r[("invalid_rating", "rating")] == 1
    assert r[("invalid_timestamp", "timestamp")] == 1
    assert r[("referential_integrity", "user_id")] == 1
    assert r[("referential_integrity", "movie_id")] == 1
    # only the first row and the last row are fully valid
    assert (~eng.error_mask).sum() == 2


def test_user_rules_warn_but_keep_rows():
    df = pd.DataFrame({"user_id": ["1", "2", "3"], "gender": ["m", "", "F"], "age": ["25", "200", "18"],
                       "occupation": ["1", "2", "3"], "zipcode": ["12345", "", "54321"]})
    eng = validate_users(df, V)
    r = _rules(eng)
    assert r[("non_standard_format", "gender")] == 1
    assert r[("missing_value", "gender")] == 1
    assert r[("invalid_domain", "age")] == 1
    assert eng.error_mask.sum() == 0          # warnings never drop rows

"""Stage 2 - Automated data validation.

Two layers:

1. **Row-level rule engine** (pandas). Every rule tags offending rows with a
   reason code and a severity:
     * ``error``   -> row is removed and written to ``data/errors/<table>_errors.csv``
     * ``warning`` -> row is kept, the issue is repaired in the preparation stage
   Checks: missing values, duplicates, incorrect data types, invalid ratings,
   invalid timestamps, domain values (gender/age/occupation) and referential
   integrity (interactions -> users / items).

2. **Schema gate** (Pandera). The cleaned output must satisfy a strict
   declarative schema; if it does not the pipeline stops.

Outputs: ``data/processed/validated_*.parquet``, ``data/errors/*``,
``reports/validation_report.md`` and ``reports/validation_report.json``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.utils import (df_to_markdown, get_logger, path, read_json, timed, write_json,
                       write_parquet)

log = get_logger("validation")


class RuleEngine:
    """Collects (rule, severity, mask) results for one table."""

    def __init__(self, table: str, df: pd.DataFrame):
        self.table, self.df = table, df
        self.results: list[dict] = []
        self.reasons = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
        self.error_mask = pd.Series(False, index=df.index)

    def check(self, rule: str, column: str, severity: str, mask: pd.Series, description: str):
        mask = mask.fillna(False).astype(bool)
        n = int(mask.sum())
        sample = self.df.loc[mask].head(3).to_dict("records") if n else []
        self.results.append({"table": self.table, "rule": rule, "column": column,
                             "severity": severity, "failed_rows": n,
                             "failed_pct": round(100 * n / max(len(self.df), 1), 4),
                             "description": description, "sample": sample})
        for i in mask[mask].index:
            self.reasons.at[i] = self.reasons.at[i] + [f"{severity}:{rule}({column})"]
        if severity == "error":
            self.error_mask |= mask
        log.info("%-13s %-26s %-9s %-7s failed=%d", self.table, rule, column, severity, n)


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def _non_numeric(s: pd.Series) -> pd.Series:
    return ~_blank(s) & pd.to_numeric(s, errors="coerce").isna()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ------------------------------------------------------------ users ----
def validate_users(df: pd.DataFrame, v: dict) -> RuleEngine:
    r = RuleEngine("users", df)
    r.check("missing_value", "user_id", "error", _blank(df.user_id), "user_id is empty")
    r.check("invalid_type", "user_id", "error", _non_numeric(df.user_id), "user_id not an integer")
    r.check("exact_duplicate", "*", "error", df.duplicated(keep="first"), "row repeated verbatim")
    r.check("duplicate_key", "user_id", "error",
            df.user_id.duplicated(keep="first") & ~df.duplicated(keep="first"),
            "same user_id with different attributes")
    r.check("missing_value", "gender", "warning", _blank(df.gender), "gender empty -> imputed 'U'")
    g = df.gender.astype(str).str.strip()
    r.check("non_standard_format", "gender", "warning",
            ~_blank(df.gender) & ~g.isin(v["valid_genders"]) & g.str.upper().isin(v["valid_genders"]),
            "gender not upper-case -> standardised")
    r.check("invalid_domain", "gender", "warning",
            ~_blank(df.gender) & ~g.str.upper().isin(v["valid_genders"]), "gender outside {M,F}")
    age = _to_num(df.age)
    r.check("invalid_domain", "age", "warning", ~age.isin(v["valid_ages"]),
            "age not a MovieLens age bucket -> imputed with mode")
    occ = _to_num(df.occupation)
    lo, hi = v["occupation_range"]
    r.check("invalid_domain", "occupation", "warning", ~occ.between(lo, hi), "occupation code out of range")
    r.check("missing_value", "zipcode", "warning", _blank(df.zipcode), "zipcode empty -> 'unknown'")
    return r


# ------------------------------------------------------------ items ----
def validate_items(df: pd.DataFrame, v: dict) -> RuleEngine:
    r = RuleEngine("items", df)
    r.check("missing_value", "movie_id", "error", _blank(df.movie_id), "movie_id is empty")
    r.check("invalid_type", "movie_id", "error", _non_numeric(df.movie_id), "movie_id not an integer")
    r.check("exact_duplicate", "*", "error", df.duplicated(keep="first"), "row repeated verbatim")
    r.check("duplicate_key", "movie_id", "error",
            df.movie_id.duplicated(keep="first") & ~df.duplicated(keep="first"),
            "same movie_id with different attributes")
    r.check("missing_value", "title", "warning", _blank(df.title), "title empty -> placeholder")
    r.check("non_standard_format", "title", "warning",
            ~_blank(df.title) & (df.title != df.title.str.strip()), "leading/trailing whitespace")
    r.check("missing_value", "genres", "warning", _blank(df.genres), "genres empty -> '(no genres listed)'")
    return r


# ----------------------------------------------------- interactions ----
def validate_interactions(df: pd.DataFrame, v: dict, valid_users: set, valid_items: set) -> RuleEngine:
    r = RuleEngine("interactions", df)
    for col in ["user_id", "movie_id", "rating", "timestamp"]:
        r.check("missing_value", col, "error", _blank(df[col]), f"{col} is empty")
    for col in ["user_id", "movie_id", "rating"]:
        r.check("invalid_type", col, "error", _non_numeric(df[col]), f"{col} is not numeric")

    rating = _to_num(df.rating)
    bad_rating = rating.notna() & ~rating.between(v["rating_min"], v["rating_max"])
    if v["rating_integer_only"]:
        bad_rating |= rating.notna() & (rating % 1 != 0)
    r.check("invalid_rating", "rating", "error", bad_rating,
            f"rating outside [{v['rating_min']},{v['rating_max']}] or not an integer")

    ts = _to_num(df.timestamp)
    tmin = pd.Timestamp(v["timestamp_min"], tz="UTC").timestamp()
    tmax = pd.Timestamp(v["timestamp_max"], tz="UTC").timestamp()
    r.check("invalid_timestamp", "timestamp", "error",
            (~_blank(df.timestamp) & ts.isna()) | (ts.notna() & ~ts.between(tmin, tmax)),
            f"timestamp malformed or outside [{v['timestamp_min']}, {v['timestamp_max']})")

    r.check("exact_duplicate", "*", "error", df.duplicated(keep="first"), "row repeated verbatim")

    uid, mid = _to_num(df.user_id), _to_num(df.movie_id)
    r.check("referential_integrity", "user_id", "error", uid.notna() & ~uid.isin(valid_users),
            "user_id not present in validated users table")
    r.check("referential_integrity", "movie_id", "error", mid.notna() & ~mid.isin(valid_items),
            "movie_id not present in validated items table")

    clean_key = ~r.error_mask
    key_dup = df[clean_key].duplicated(subset=["user_id", "movie_id"], keep=False)
    r.check("duplicate_key", "user_id+movie_id", "warning", key_dup.reindex(df.index, fill_value=False),
            "user re-rated the same item -> latest kept in preparation")
    return r


# -------------------------------------------------------- pandera gate ----
def schemas(v: dict) -> dict[str, pa.DataFrameSchema]:
    tmin = int(pd.Timestamp(v["timestamp_min"], tz="UTC").timestamp())
    tmax = int(pd.Timestamp(v["timestamp_max"], tz="UTC").timestamp())
    return {
        "users": pa.DataFrameSchema({
            "user_id": pa.Column(int, pa.Check.gt(0), unique=True),
            "gender": pa.Column(str, nullable=True),
            "age": pa.Column(float, nullable=True),
            "occupation": pa.Column(float, nullable=True),
            "zipcode": pa.Column(str, nullable=True),
        }, strict=True),
        "items": pa.DataFrameSchema({
            "movie_id": pa.Column(int, pa.Check.gt(0), unique=True),
            "title": pa.Column(str, nullable=True),
            "genres": pa.Column(str, nullable=True),
        }, strict=True),
        "interactions": pa.DataFrameSchema({
            "user_id": pa.Column(int, pa.Check.gt(0)),
            "movie_id": pa.Column(int, pa.Check.gt(0)),
            "rating": pa.Column(int, pa.Check.in_range(v["rating_min"], v["rating_max"])),
            "timestamp": pa.Column(int, pa.Check.in_range(tmin, tmax)),
        }, strict=True),
    }


def _split(cfg: dict, r: RuleEngine, casts: dict) -> pd.DataFrame:
    df = r.df
    errors = df[r.error_mask].copy()
    errors["error_reasons"] = r.reasons[r.error_mask].map("; ".join)
    errors.to_csv(path(cfg, "errors", f"{r.table}_errors.csv"), index=False)

    warn = df[~r.error_mask & r.reasons.map(len).gt(0)].copy()
    warn["warning_reasons"] = r.reasons[warn.index].map("; ".join)
    warn.to_csv(path(cfg, "errors", f"{r.table}_warnings.csv"), index=False)

    valid = df[~r.error_mask].copy()
    for col, kind in casts.items():
        if kind == "int":
            valid[col] = _to_num(valid[col]).astype("int64")
        elif kind == "float":
            valid[col] = _to_num(valid[col]).astype("float64")
        else:
            valid[col] = valid[col].replace("", np.nan).astype(object)
    return valid.reset_index(drop=True)


def run(cfg: dict) -> dict:
    v = cfg["validation"]
    with timed(log, "validation"):
        raw = {t: pd.read_csv(path(cfg, "raw", f"{t}.csv"), dtype=str, keep_default_na=False)
               for t in ["users", "items", "interactions"]}

        ru = validate_users(raw["users"], v)
        users = _split(cfg, ru, {"user_id": "int", "gender": "str", "age": "float",
                                 "occupation": "float", "zipcode": "str"})
        users.loc[~users.age.isin(v["valid_ages"]), "age"] = np.nan

        ri = validate_items(raw["items"], v)
        items = _split(cfg, ri, {"movie_id": "int", "title": "str", "genres": "str"})

        rx = validate_interactions(raw["interactions"], v, set(users.user_id), set(items.movie_id))
        inter = _split(cfg, rx, {"user_id": "int", "movie_id": "int", "rating": "int", "timestamp": "int"})

        gate = {}
        for name, df, sch in [("users", users, None), ("items", items, None), ("interactions", inter, None)]:
            try:
                schemas(v)[name].validate(df, lazy=True)
                gate[name] = "PASSED"
            except pa.errors.SchemaErrors as e:
                gate[name] = f"FAILED: {len(e.failure_cases)} failure cases"
            log.info("pandera schema gate [%s]: %s", name, gate[name])

        write_parquet(users, path(cfg, "processed", "validated_users.parquet"))
        write_parquet(items, path(cfg, "processed", "validated_items.parquet"))
        write_parquet(inter, path(cfg, "processed", "validated_interactions.parquet"))

        engines = [ru, ri, rx]
        summary = []
        for e in engines:
            n = len(e.df)
            n_err = int(e.error_mask.sum())
            n_warn = int((~e.error_mask & e.reasons.map(len).gt(0)).sum())
            summary.append({"table": e.table, "input_rows": n, "rejected_rows": n_err,
                            "rows_with_warnings": n_warn, "valid_rows": n - n_err,
                            "error_rate_pct": round(100 * n_err / n, 3),
                            "schema_gate": gate[e.table]})
        report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "summary": summary,
                  "checks": [c for e in engines for c in e.results]}
        write_json(report, path(cfg, "reports", "validation_report.json"))
        _write_markdown(cfg, report)

        worst = max(s["error_rate_pct"] for s in summary) / 100
        if worst > v["fail_pipeline_if_error_rate_above"]:
            raise RuntimeError(f"Quality gate failed: error rate {worst:.2%} above threshold")
        if any(g != "PASSED" for g in gate.values()):
            raise RuntimeError(f"Schema gate failed: {gate}")
    return report


def _write_markdown(cfg: dict, report: dict) -> None:
    summ = pd.DataFrame(report["summary"])
    checks = pd.DataFrame(report["checks"]).drop(columns=["sample"])
    checks = checks[checks.failed_rows > 0] if len(checks) else checks
    all_checks = pd.DataFrame(report["checks"])[["table", "rule", "column", "severity", "failed_rows"]]

    recon = ""
    manifest_p = path(cfg, "raw", "ingestion_manifest.json")
    if manifest_p.exists():
        m = read_json(manifest_p)
        if m.get("anomaly_injection_enabled"):
            rows = [{"table": t, "anomaly": k, "injected": n} for t, d in m["injected_anomalies"].items()
                    for k, n in d.items()]
            recon = ("\n## Reconciliation with injected anomalies\n\n"
                     "Anomalies deliberately injected at ingestion (seeded, see config). Detected counts "
                     "in the table above can differ slightly because a few rows received two anomalies "
                     "(e.g. an orphan id *and* a missing rating) and because MovieLens contains no natural "
                     "errors.\n\n" + df_to_markdown(pd.DataFrame(rows)) + "\n")

    md = f"""# Data Validation Report

Generated: {report['generated_at_utc']}

Severity policy: **error** rows are quarantined to `data/errors/<table>_errors.csv` (with an
`error_reasons` column); **warning** rows are kept and repaired in the preparation stage
(listed in `data/errors/<table>_warnings.csv`). After the rule engine, the clean tables must
pass a strict Pandera schema gate.

## Summary

{df_to_markdown(summ, '.3f')}

## Checks that found issues

{df_to_markdown(checks[['table','rule','column','severity','failed_rows','failed_pct','description']], '.4f')}

## Full check catalogue (including checks that passed)

{df_to_markdown(all_checks)}
{recon}"""
    path(cfg, "reports", "validation_report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    from src.utils import load_config
    run(load_config())

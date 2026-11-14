"""Stage 3 - Data preparation (cleaning & standardisation).

Input : ``data/processed/validated_*.parquet`` (rows that passed validation)
Output: ``data/processed/{users,items,interactions}.parquet`` (clean, typed)

Cleaning actions (each one is counted and written to ``reports/preparation_log.json``):

* users  - gender upper-cased, missing gender -> "U"; invalid/missing age -> mode bucket;
           age/occupation descriptions attached; zipcode normalised to 5-char string or "unknown".
* items  - titles stripped of whitespace, missing title -> placeholder; release year
           extracted from title; missing genres -> "(no genres listed)"; genres split to a list.
* interactions - re-ratings of the same (user, item) resolved by keeping the latest one;
           unix timestamp converted to timezone-aware UTC datetime; sorted by time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger, path, read_parquet, timed, write_json, write_parquet

log = get_logger("preparation")

AGE_DESC = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCC_DESC = {0: "other/not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
            4: "college/grad student", 5: "customer service", 6: "doctor/health care",
            7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
            11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist",
            16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman",
            19: "unemployed", 20: "writer"}
GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]


def prepare_users(df: pd.DataFrame, actions: dict) -> pd.DataFrame:
    df = df.copy()
    g = df.gender.astype("string").str.strip().str.upper()
    actions["users.gender_standardised"] = int((g != df.gender.astype("string")).fillna(False).sum())
    actions["users.gender_imputed_U"] = int(g.isna().sum())
    df["gender"] = g.fillna("U")

    mode_age = df.age.mode().iloc[0]
    actions["users.age_imputed_mode"] = int(df.age.isna().sum())
    df["age"] = df.age.fillna(mode_age).astype(int)
    df["age_desc"] = df.age.map(AGE_DESC)

    actions["users.occupation_imputed_0"] = int(df.occupation.isna().sum())
    df["occupation"] = df.occupation.fillna(0).astype(int)
    df["occ_desc"] = df.occupation.map(OCC_DESC)

    z = df.zipcode.astype("string").str.strip().str.extract(r"^(\d{5})", expand=False)
    actions["users.zipcode_unknown"] = int(z.isna().sum())
    df["zipcode"] = z.fillna("unknown")
    df["region"] = np.where(df.zipcode == "unknown", "unknown", df.zipcode.str[0])
    return df.sort_values("user_id").reset_index(drop=True)


def prepare_items(df: pd.DataFrame, actions: dict) -> pd.DataFrame:
    df = df.copy()
    t = df.title.astype("string")
    actions["items.title_whitespace_stripped"] = int((t != t.str.strip()).fillna(False).sum())
    t = t.str.strip()
    actions["items.title_placeholder"] = int(t.isna().sum())
    df["title"] = t.fillna("Unknown title #" + df.movie_id.astype(str))

    year = df.title.str.extract(r"\((\d{4})\)\s*$", expand=False)
    df["year"] = pd.to_numeric(year, errors="coerce").astype("Int64")
    df["clean_title"] = df.title.str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

    actions["items.genres_filled"] = int(df.genres.isna().sum())
    df["genres"] = df.genres.fillna("(no genres listed)")
    df["genre_list"] = df.genres.str.split("|")
    df["primary_genre"] = df.genre_list.str[0]
    return df.sort_values("movie_id").reset_index(drop=True)


def prepare_interactions(df: pd.DataFrame, actions: dict, keep: str) -> pd.DataFrame:
    df = df.sort_values(["user_id", "movie_id", "timestamp"])
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep=keep)
    actions["interactions.re_ratings_resolved"] = before - len(df)
    df = df.rename(columns={"timestamp": "unix_ts"})
    df["timestamp"] = pd.to_datetime(df.unix_ts, unit="s", utc=True)
    df["rating"] = df.rating.astype("int8")
    return df.sort_values("timestamp").reset_index(drop=True)


def run(cfg: dict) -> dict:
    actions: dict[str, int] = {}
    with timed(log, "preparation"):
        users = prepare_users(read_parquet(path(cfg, "processed", "validated_users.parquet")), actions)
        items = prepare_items(read_parquet(path(cfg, "processed", "validated_items.parquet")), actions)
        inter = prepare_interactions(read_parquet(path(cfg, "processed", "validated_interactions.parquet")),
                                     actions, cfg["preparation"]["dedup_interactions_keep"])
        write_parquet(users, path(cfg, "processed", "users.parquet"))
        write_parquet(items, path(cfg, "processed", "items.parquet"))
        write_parquet(inter, path(cfg, "processed", "interactions.parquet"))
        out = {"actions": actions,
               "rows": {"users": len(users), "items": len(items), "interactions": len(inter)}}
        for k, val in actions.items():
            log.info("%-40s %d", k, val)
        log.info("clean rows: %s", out["rows"])
        write_json(out, path(cfg, "reports", "preparation_log.json"))
    return out


if __name__ == "__main__":
    from src.utils import load_config
    run(load_config())

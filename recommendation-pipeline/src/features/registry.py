"""Feature registry - the single source of truth for every feature.

Each entry documents name, entity, dtype, description and how it is derived.
The registry is exported to ``data/features/feature_registry.json`` and
rendered as the feature catalog ``reports/feature_catalog.md``.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureDef:
    name: str
    entity: str          # user | item | user_genre | user_item
    dtype: str
    description: str
    derivation: str
    used_by: str = "-"   # models / serving that consume the feature


FEATURES: list[FeatureDef] = [
    # ------------------------------------------------------------- user ----
    FeatureDef("user_activity_count", "user", "int", "Number of ratings the user gave before the cutoff",
               "count(ratings) per user_id on train", "Hybrid, analysis"),
    FeatureDef("user_avg_rating", "user", "float", "Mean rating given by the user",
               "mean(rating) per user_id", "MF (bias init), Hybrid"),
    FeatureDef("user_rating_std", "user", "float", "Std-dev of the user's ratings (0 if one rating)",
               "std(rating) per user_id", "analysis"),
    FeatureDef("user_rating_bias", "user", "float", "How much more generous the user is than average",
               "user_avg_rating - global_mean_rating", "analysis"),
    FeatureDef("user_like_ratio", "user", "float", "Share of the user's ratings that are >= 4",
               "mean(rating >= 4) per user_id", "analysis"),
    FeatureDef("user_days_since_last_interaction", "user", "int", "Recency: days from last rating to cutoff",
               "(cutoff - max(timestamp)).days", "serving (recency filter)"),
    FeatureDef("user_tenure_days", "user", "int", "Days between first and last rating",
               "(max(ts) - min(ts)).days + 1", "analysis"),
    FeatureDef("user_n_distinct_genres", "user", "int", "Breadth of taste: distinct genres rated",
               "nunique(genre) over exploded genre_list", "analysis"),
    FeatureDef("user_preferred_genre", "user", "category", "Genre the user rated most often",
               "argmax(count) over user's genres", "serving (explanations)"),
    FeatureDef("user_mainstreamness", "user", "float", "Avg popularity percentile of items the user rated",
               "mean(item_popularity_percentile) over user's items", "analysis"),
    FeatureDef("user_gender", "user", "category", "Gender (M/F/U after imputation)", "users.gender", "analysis"),
    FeatureDef("user_age_bucket", "user", "int", "MovieLens age bucket", "users.age", "analysis"),
    FeatureDef("user_occupation", "user", "int", "Occupation code 0-20", "users.occupation", "analysis"),
    # ------------------------------------------------------------- item ----
    FeatureDef("item_popularity", "item", "int", "Number of ratings received before the cutoff",
               "count(ratings) per movie_id", "Popularity, Hybrid (prior)"),
    FeatureDef("item_popularity_percentile", "item", "float", "Popularity percentile rank (1 = most popular)",
               "rank(item_popularity, pct=True)", "user_mainstreamness"),
    FeatureDef("item_avg_rating", "item", "float", "Mean rating received", "mean(rating) per movie_id",
               "rating baseline"),
    FeatureDef("item_bayesian_avg", "item", "float", "Rating shrunk towards the global mean",
               "(n*avg + m*global_mean)/(n+m), m = bayesian_prior_count", "Popularity (alt. score)"),
    FeatureDef("item_rating_std", "item", "float", "Std-dev of ratings received", "std(rating) per movie_id",
               "analysis"),
    FeatureDef("item_like_ratio", "item", "float", "Share of ratings >= 4", "mean(rating >= 4) per movie_id",
               "analysis"),
    FeatureDef("item_days_since_last_interaction", "item", "int", "Days from the item's last rating to cutoff",
               "(cutoff - max(timestamp)).days", "analysis"),
    FeatureDef("item_release_year", "item", "int", "Release year parsed from title", "regex on title",
               "analysis"),
    FeatureDef("item_age_years", "item", "int", "Years between release and cutoff",
               "cutoff.year - item_release_year", "analysis"),
    FeatureDef("item_primary_genre", "item", "category", "First listed genre", "genre_list[0]", "analysis"),
    FeatureDef("genre_<name> (x18)", "item", "int(0/1)", "Multi-hot genre indicators",
               "one-hot of genre_list", "Hybrid, Diversity metric"),
    # ------------------------------------------------------- user_genre ----
    FeatureDef("user_genre_share", "user_genre", "float", "Fraction of the user's ratings in this genre",
               "count(user,genre)/count(user)", "Hybrid"),
    FeatureDef("user_genre_avg_rating", "user_genre", "float", "User's mean rating within the genre",
               "mean(rating) per (user, genre)", "Hybrid"),
    FeatureDef("user_genre_affinity", "user_genre", "float", "Combined preference score for the genre",
               "user_genre_share * user_genre_avg_rating / 5", "Hybrid"),
    # -------------------------------------------------------- user_item ----
    FeatureDef("ui_genre_affinity", "user_item", "float",
               "On-demand: mean user_genre_affinity over the candidate item's genres",
               "computed by FeatureStore.get_user_item_features()", "Hybrid (train + serve)"),
    FeatureDef("ui_popularity_gap", "user_item", "float",
               "On-demand: item popularity percentile minus user mainstreamness",
               "computed by FeatureStore.get_user_item_features()", "analysis / re-ranking"),
]


def as_records() -> list[dict]:
    return [asdict(f) for f in FEATURES]

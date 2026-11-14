# Feature Catalog

Feature version **v1**, computed as of the train/test cutoff `2000-12-02T14:58:06+00:00` from training data only.

| Feature group | Entity key | Rows | File |
|---|---|---|---|
| User features | user_id | 5400 | data/features/user_features.parquet |
| Item features | movie_id | 3883 | data/features/item_features.parquet |
| User-genre features | user_id, genre | 82782 | data/features/user_genre_features.parquet |
| User-item features | user_id, movie_id | on demand | FeatureStore.get_user_item_features() |

## Feature definitions (29 entries)

| name | entity | dtype | description | derivation | used_by |
| --- | --- | --- | --- | --- | --- |
| user_activity_count | user | int | Number of ratings the user gave before the cutoff | count(ratings) per user_id on train | Hybrid, analysis |
| user_avg_rating | user | float | Mean rating given by the user | mean(rating) per user_id | MF (bias init), Hybrid |
| user_rating_std | user | float | Std-dev of the user's ratings (0 if one rating) | std(rating) per user_id | analysis |
| user_rating_bias | user | float | How much more generous the user is than average | user_avg_rating - global_mean_rating | analysis |
| user_like_ratio | user | float | Share of the user's ratings that are >= 4 | mean(rating >= 4) per user_id | analysis |
| user_days_since_last_interaction | user | int | Recency: days from last rating to cutoff | (cutoff - max(timestamp)).days | serving (recency filter) |
| user_tenure_days | user | int | Days between first and last rating | (max(ts) - min(ts)).days + 1 | analysis |
| user_n_distinct_genres | user | int | Breadth of taste: distinct genres rated | nunique(genre) over exploded genre_list | analysis |
| user_preferred_genre | user | category | Genre the user rated most often | argmax(count) over user's genres | serving (explanations) |
| user_mainstreamness | user | float | Avg popularity percentile of items the user rated | mean(item_popularity_percentile) over user's items | analysis |
| user_gender | user | category | Gender (M/F/U after imputation) | users.gender | analysis |
| user_age_bucket | user | int | MovieLens age bucket | users.age | analysis |
| user_occupation | user | int | Occupation code 0-20 | users.occupation | analysis |
| item_popularity | item | int | Number of ratings received before the cutoff | count(ratings) per movie_id | Popularity, Hybrid (prior) |
| item_popularity_percentile | item | float | Popularity percentile rank (1 = most popular) | rank(item_popularity, pct=True) | user_mainstreamness |
| item_avg_rating | item | float | Mean rating received | mean(rating) per movie_id | rating baseline |
| item_bayesian_avg | item | float | Rating shrunk towards the global mean | (n*avg + m*global_mean)/(n+m), m = bayesian_prior_count | Popularity (alt. score) |
| item_rating_std | item | float | Std-dev of ratings received | std(rating) per movie_id | analysis |
| item_like_ratio | item | float | Share of ratings >= 4 | mean(rating >= 4) per movie_id | analysis |
| item_days_since_last_interaction | item | int | Days from the item's last rating to cutoff | (cutoff - max(timestamp)).days | analysis |
| item_release_year | item | int | Release year parsed from title | regex on title | analysis |
| item_age_years | item | int | Years between release and cutoff | cutoff.year - item_release_year | analysis |
| item_primary_genre | item | category | First listed genre | genre_list[0] | analysis |
| genre_<name> (x18) | item | int(0/1) | Multi-hot genre indicators | one-hot of genre_list | Hybrid, Diversity metric |
| user_genre_share | user_genre | float | Fraction of the user's ratings in this genre | count(user,genre)/count(user) | Hybrid |
| user_genre_avg_rating | user_genre | float | User's mean rating within the genre | mean(rating) per (user, genre) | Hybrid |
| user_genre_affinity | user_genre | float | Combined preference score for the genre | user_genre_share * user_genre_avg_rating / 5 | Hybrid |
| ui_genre_affinity | user_item | float | On-demand: mean user_genre_affinity over the candidate item's genres | computed by FeatureStore.get_user_item_features() | Hybrid (train + serve) |
| ui_popularity_gap | user_item | float | On-demand: item popularity percentile minus user mainstreamness | computed by FeatureStore.get_user_item_features() | analysis / re-ranking |

## Reuse for training and prediction

All models and the serving script read features through `src.features.build_features.FeatureStore`, never by recomputing them. Training reads them in `src/training/models.py` (Popularity -> `item_popularity`, Hybrid -> `user_genre_affinity`, `genre_*`, `item_popularity`). Prediction reads them in `src/serving/recommend.py`.

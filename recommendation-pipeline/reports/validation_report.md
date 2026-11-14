# Data Validation Report

Generated: 2026-11-14T03:05:40.149376+00:00

Severity policy: **error** rows are quarantined to `data/errors/<table>_errors.csv` (with an
`error_reasons` column); **warning** rows are kept and repaired in the preparation stage
(listed in `data/errors/<table>_warnings.csv`). After the rule engine, the clean tables must
pass a strict Pandera schema gate.

## Summary

| table | input_rows | rejected_rows | rows_with_warnings | valid_rows | error_rate_pct | schema_gate |
| --- | --- | --- | --- | --- | --- | --- |
| users | 6060 | 20 | 65 | 6040 | 0.330 | PASSED |
| items | 3893 | 10 | 33 | 3883 | 0.257 | PASSED |
| interactions | 1001509 | 2598 | 600 | 998911 | 0.259 | PASSED |

## Checks that found issues

| table | rule | column | severity | failed_rows | failed_pct | description |
| --- | --- | --- | --- | --- | --- | --- |
| users | exact_duplicate | * | error | 19 | 0.3135 | row repeated verbatim |
| users | duplicate_key | user_id | error | 1 | 0.0165 | same user_id with different attributes |
| users | missing_value | gender | warning | 15 | 0.2475 | gender empty -> imputed 'U' |
| users | non_standard_format | gender | warning | 25 | 0.4125 | gender not upper-case -> standardised |
| users | invalid_domain | age | warning | 10 | 0.1650 | age not a MovieLens age bucket -> imputed with mode |
| users | missing_value | zipcode | warning | 15 | 0.2475 | zipcode empty -> 'unknown' |
| items | exact_duplicate | * | error | 9 | 0.2312 | row repeated verbatim |
| items | duplicate_key | movie_id | error | 1 | 0.0257 | same movie_id with different attributes |
| items | missing_value | title | warning | 5 | 0.1284 | title empty -> placeholder |
| items | non_standard_format | title | warning | 20 | 0.5137 | leading/trailing whitespace |
| items | missing_value | genres | warning | 8 | 0.2055 | genres empty -> '(no genres listed)' |
| interactions | missing_value | user_id | error | 200 | 0.0200 | user_id is empty |
| interactions | missing_value | rating | error | 300 | 0.0300 | rating is empty |
| interactions | invalid_type | rating | error | 200 | 0.0200 | rating is not numeric |
| interactions | invalid_rating | rating | error | 300 | 0.0300 | rating outside [1,5] or not an integer |
| interactions | invalid_timestamp | timestamp | error | 300 | 0.0300 | timestamp malformed or outside [1995-01-01, 2026-01-01) |
| interactions | exact_duplicate | * | error | 999 | 0.0997 | row repeated verbatim |
| interactions | referential_integrity | user_id | error | 150 | 0.0150 | user_id not present in validated users table |
| interactions | referential_integrity | movie_id | error | 150 | 0.0150 | movie_id not present in validated items table |
| interactions | duplicate_key | user_id+movie_id | warning | 600 | 0.0599 | user re-rated the same item -> latest kept in preparation |

## Full check catalogue (including checks that passed)

| table | rule | column | severity | failed_rows |
| --- | --- | --- | --- | --- |
| users | missing_value | user_id | error | 0 |
| users | invalid_type | user_id | error | 0 |
| users | exact_duplicate | * | error | 19 |
| users | duplicate_key | user_id | error | 1 |
| users | missing_value | gender | warning | 15 |
| users | non_standard_format | gender | warning | 25 |
| users | invalid_domain | gender | warning | 0 |
| users | invalid_domain | age | warning | 10 |
| users | invalid_domain | occupation | warning | 0 |
| users | missing_value | zipcode | warning | 15 |
| items | missing_value | movie_id | error | 0 |
| items | invalid_type | movie_id | error | 0 |
| items | exact_duplicate | * | error | 9 |
| items | duplicate_key | movie_id | error | 1 |
| items | missing_value | title | warning | 5 |
| items | non_standard_format | title | warning | 20 |
| items | missing_value | genres | warning | 8 |
| interactions | missing_value | user_id | error | 200 |
| interactions | missing_value | movie_id | error | 0 |
| interactions | missing_value | rating | error | 300 |
| interactions | missing_value | timestamp | error | 0 |
| interactions | invalid_type | user_id | error | 0 |
| interactions | invalid_type | movie_id | error | 0 |
| interactions | invalid_type | rating | error | 200 |
| interactions | invalid_rating | rating | error | 300 |
| interactions | invalid_timestamp | timestamp | error | 300 |
| interactions | exact_duplicate | * | error | 999 |
| interactions | referential_integrity | user_id | error | 150 |
| interactions | referential_integrity | movie_id | error | 150 |
| interactions | duplicate_key | user_id+movie_id | warning | 600 |

## Reconciliation with injected anomalies

Anomalies deliberately injected at ingestion (seeded, see config). Detected counts in the table above can differ slightly because a few rows received two anomalies (e.g. an orphan id *and* a missing rating) and because MovieLens contains no natural errors.

| table | anomaly | injected |
| --- | --- | --- |
| interactions | exact_duplicates | 1000 |
| interactions | re_ratings | 300 |
| interactions | missing_rating | 300 |
| interactions | missing_user_id | 200 |
| interactions | out_of_range_rating | 300 |
| interactions | non_numeric_rating | 200 |
| interactions | invalid_timestamp | 300 |
| interactions | orphan_user | 150 |
| interactions | orphan_item | 150 |
| users | exact_duplicates | 20 |
| users | missing_gender | 15 |
| users | lowercase_gender | 25 |
| users | invalid_age | 10 |
| users | missing_zipcode | 15 |
| items | exact_duplicates | 10 |
| items | missing_genres | 8 |
| items | missing_title | 5 |
| items | padded_title | 20 |

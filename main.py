from src.eda.extract import extract_files_from_zipfile
from src.eda.dataset_shape import count_rows_and_columns
from src.eda.datatype_distribution import get_column_names_and_dtypes
from src.eda.missing_data_analysis import get_null_count_per_column
from src.eda.data_quality_report import get_data_quality_for_authors
from src.eda.data_quality_report import get_data_quality_for_tweets
from src.eda.univariate_analysis import get_univariate_for_authors
from src.eda.univariate_analysis import get_univariate_for_tweets
from src.eda.univariate_analysis import get_univariate_for_tweet_categoricals
from src.eda.univariate_analysis import get_temporal_distribution

extract_files_from_zipfile("data/archive.zip")

# ── Number of rows and columns ─────────────────────────────────────────────
twitter_users_column_and_row_count = count_rows_and_columns(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
flood_control_tweets_column_and_row_count = count_rows_and_columns(
    "data/for_export_dpwh_floodcontrol.csv"
)

# ── Column names and data types ────────────────────────────────────────────
twitter_users_columns_and_dtypes = get_column_names_and_dtypes(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
flood_control_tweets_columns_and_dtypes = get_column_names_and_dtypes(
    "data/for_export_dpwh_floodcontrol.csv"
)

# ── Null count and null count percentage ───────────────────────────────────
twitter_users_null_count_per_column = get_null_count_per_column(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
flood_control_tweets_null_count_per_column = get_null_count_per_column(
    "data/for_export_dpwh_floodcontrol.csv"
)

# ── Data quality report ────────────────────────────────────────────────────
get_data_quality_for_authors(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
get_data_quality_for_tweets(
    "data/for_export_dpwh_floodcontrol.csv"
)

# ── Univariate analysis — Dataset: Authors ──────────────────────────────
get_univariate_for_authors(
    "data/well_known_authors_dpwh_floodcontrol.csv",
    save_path="output/well_known_authors_dpwh_floodcontrol_author_distribution.png"
)

# ── Univariate analysis — Dataset: Tweets ───────────────────────────────
get_univariate_for_tweets(
    "data/for_export_dpwh_floodcontrol.csv",
    save_path="output/for_export_dpwh_floodcontrol_engagement_distribution.png"
)
get_univariate_for_tweet_categoricals(
    "data/for_export_dpwh_floodcontrol.csv",
    save_path="output/for_export_dpwh_floodcontrol_categorical_distribution.png"
)
get_temporal_distribution(
    "data/for_export_dpwh_floodcontrol.csv",
    save_path="output/for_export_dpwh_floodcontrol_temporal_distribution.png"
)
from src.eda import extract_files_from_zipfile
from src.eda import count_rows_and_columns
from src.eda import get_column_names_and_dtypes
from src.eda import display_first_few_rows
from src.eda import get_null_count_per_column
from src.eda import get_duplicates_count_per_column

extract_files_from_zipfile("data/archive.zip")

# Number of rows and columns
twitter_users_column_and_row_count = count_rows_and_columns(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
flood_control_tweets_column_and_row_count = count_rows_and_columns(
    "data/for_export_dpwh_floodcontrol.csv"
)

# Column names and data types
twitter_users_columns_and_dtypes = get_column_names_and_dtypes(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
flood_control_tweets_columns_and_dtypes = get_column_names_and_dtypes(
    "data/for_export_dpwh_floodcontrol.csv"
)

# Preview of first few rows
twitter_users_preview = display_first_few_rows(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
flood_control_tweets_preview = display_first_few_rows(
    "data/for_export_dpwh_floodcontrol.csv"    
)

# Null count and null count percentage
twitter_users_null_count_per_column = get_null_count_per_column(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
flood_control_tweets_null_count_per_column = get_null_count_per_column(
    "data/for_export_dpwh_floodcontrol.csv"
)

twitter_users_duplicates_count_per_column = get_duplicates_count_per_column(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
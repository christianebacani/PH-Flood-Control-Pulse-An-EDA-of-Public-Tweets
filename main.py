from src.eda import extract_files_from_zipfile
from src.eda import count_rows_and_columns
from src.eda import get_column_names_and_dtypes
from src.eda import display_first_few_rows

extract_files_from_zipfile("data/archive.zip")

twitter_authors_column_and_row_count = count_rows_and_columns(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
dpwh_flood_control_tweets_column_and_row_count = count_rows_and_columns(
    "data/for_export_dpwh_floodcontrol.csv"
)

twitter_authors_columns_and_dtypes = get_column_names_and_dtypes(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
dpwh_flood_control_tweets_columns_and_dtypes = get_column_names_and_dtypes(
    "data/for_export_dpwh_floodcontrol.csv"
)

display_first_few_rows(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
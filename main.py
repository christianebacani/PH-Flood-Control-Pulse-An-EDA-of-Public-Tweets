from src import eda

eda.extract_files_from_zipfile("data/archive.zip")

twitter_authors_column_and_row_count = eda.count_rows_and_columns(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
dphwh_flood_control_tweets_column_and_row_count = eda.count_rows_and_columns(
    "data/for_export_dpwh_floodcontrol.csv"
)

twitter_authors_columns_and_dtypes = eda.get_column_names_and_dtypes(
    "data/well_known_authors_dpwh_floodcontrol.csv"
)
import pandas as pd
from zipfile import ZipFile

def extract_files_from_zipfile(filepath: str) -> None:
    # Extract all the files inside the zipfile of data/ directory
    with ZipFile("data/archive.zip", 'r') as zip_file:
        zip_file.extractall("data") # File destination

def count_rows_and_columns(filepath: str) -> list[int]:
    df = pd.read_csv(filepath)
    # Get the number of columns and rows of the dataset
    column_count = len(list(df.keys()))
    row_count = len(df)

    return [column_count, row_count]

def get_column_names_and_dtypes(filepath: str) -> pd.Series:
    df = pd.read_csv(filepath)
    # Get the column names and datatype of the dataset
    column_names_and_dtypes = df.dtypes

    return column_names_and_dtypes

def display_first_few_rows(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    first_few_rows = df.head(3)

    return first_few_rows
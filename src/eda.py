import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt

def extract_files_from_zipfile(filepath: str) -> None:
    # Extract all the files inside the zipfile of data/ directory
    with ZipFile("data/archive.zip", 'r') as zip_file:
        zip_file.extractall("data") # File destination

def count_rows_and_columns(filepath: str) -> list[int]:
    # Read dataset
    df = pd.read_csv(filepath)
    rows, cols = df.shape

    # Plot
    plt.figure(figsize=(6,5))
    plt.bar(["Rows", "Columns"], [rows, cols], color=['skyblue', 'salmon'])
    plt.title("Dataset Shape Overview")
    plt.ylabel("Count")

    # Save figure
    plt.savefig(
        "output/well_known_authors_dpwh_floodcontrol_dataset_shape.png", 
         dpi=300,
         bbox_inches='tight'
    )
    plt.close()

    return [rows, cols]

def get_column_names_and_dtypes(filepath: str) -> pd.Series:
    df = pd.read_csv(filepath)
    # Get the column names and datatype of the dataset
    column_names_and_dtypes = df.dtypes

    return column_names_and_dtypes

def display_first_few_rows(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Get the first 3 rows of the dataset
    first_few_rows = df.head(3)

    return first_few_rows
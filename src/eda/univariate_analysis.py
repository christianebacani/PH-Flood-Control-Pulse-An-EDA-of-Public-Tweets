import pandas as pd

def summary_statistics(filepath: str):
    df = pd.read_csv(filepath)
    columns = df.columns

    column_and_values = {}

    for column in columns:
        if column == "author_createdAt":
            author_createdAt = df["author_createdAt"]
            column_and_values["author_createdAt"] = author_createdAt

    year_periods = {}

    for value in column_and_values["author_createdAt"]:
        year = str(value).split("-")[0]
        year = year.strip()
        year_periods[year] = []
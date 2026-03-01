import pandas as pd

def summary_statistics(filepath: str):
    df = pd.read_csv(filepath)

    years = []

    for _, row in df.iterrows():
        author_createdAt = row["author_createdAt"]
        year = str(author_createdAt).split("-")[0].strip()
        years.append(year)

    df["years"] = years
    print(df)
import pandas as pd

def summary_statistics(filepath: str):
    df = pd.read_csv(filepath)
    columns = df.columns
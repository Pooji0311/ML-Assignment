import pandas as pd

def clean_data(df):
    # Remove missing values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    return df
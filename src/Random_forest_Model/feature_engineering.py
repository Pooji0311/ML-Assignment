#Imash
#file name - feature_engineering.py

import pandas as pd


def encode_features(df):
    # Convert target to binary
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv("../data/bank.csv", sep=';')

    from Random_forest_Model_ipynb.data_cleaning import clean_data
    df = clean_data(df)

    df = encode_features(df)

    print("Encoded dataset shape:", df.shape)
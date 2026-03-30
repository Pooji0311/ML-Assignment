import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_basic_info(df):
    print("\n--- BASIC INFO ---")
    print(df.info())

    print("\n--- DATA SHAPE ---")
    print("Rows:", df.shape[0], "Columns:", df.shape[1])

def show_statistics(df):
    print("\n--- STATISTICS ---")
    print(df.describe())

def check_missing_and_unknown(df):
    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    print(missing)

    print("\n--- UNKNOWN VALUES ---")
    unknown = (df == "unknown").sum()
    print(unknown[unknown > 0])

def plot_target_distribution(df):
    print("\n--- TARGET DISTRIBUTION ---")
    print(df['y'].value_counts())

    sns.countplot(x='y', data=df)
    plt.title("Target Distribution (y)")
    plt.show()

def plot_numerical_distributions(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_categorical_counts(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'y':
            plt.figure(figsize=(8, 4))
            sns.countplot(x=col, data=df)
            plt.xticks(rotation=45)
            plt.title(f"Count Plot of {col}")
            plt.show()



def perform_all_eda(df):
    print("\n EDA START =================")
    show_basic_info(df)
    show_statistics(df)
    check_missing_and_unknown(df)
    plot_target_distribution(df)
    plot_numerical_distributions(df)
    plot_correlation_heatmap(df)
    plot_categorical_counts(df)
    print("\n================= EDA END")

if __name__ == "__main__":
    df = pd.read_csv("../data/bank.csv", sep=';')
    perform_all_eda(df)

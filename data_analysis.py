import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

## Data Loading (train and test)
def load_data(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    return train_df, test_df

# Data Overview
def data_overview(df):
    print("Data Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nStatistical Summary:\n", df.describe())
    print("\nUnique Values:\n", df.nunique())
    print("\nHead:\n", df.head())
    print("\nTail:\n", df.tail())

# Data Visualization
def plot_missing_values(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()

## column distribution
def plot_column_distribution(df, column_name):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=column_name)
    plt.title(f"Distribution of {column_name}")
    plt.xticks(rotation=45)
    plt.show()


# Relation between Loan_status and other columns
def plot_loan_status_relation(df, column_name):
    print(pd.crosstab(df[column_name],df["Loan_Status"]))
    CreditHistory = pd.crosstab(df[column_name],df["Loan_Status"])
    CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
    plt.xlabel(column_name)
    plt.ylabel("Percentage")
    plt.show()


# Correlation Heatmap
def correlation_heatmap(train):
    matrix = train.select_dtypes(include=['number']).corr()
    #matrix = train.corr()
    f, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu",annot=True);
    plt.show()


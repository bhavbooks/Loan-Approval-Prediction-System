import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


## Data Loading (train and test)

train_file_path = "/Users/mac/Desktop/Desktop_Files/Projects/Loan Approval Prediction/data/train.csv"
test_file_path = "/Users/mac/Desktop/Desktop_Files/Projects/Loan Approval Prediction/data/test.csv"

def load_data(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    print("Data Loaded Successfully ✅")
    return train_df, test_df

train_df, test_df = load_data(train_file_path, test_file_path)

# Data Overview
def data_overview(df):
    print("Data Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nStatistical Summary:\n", df.describe())
    print("\nUnique Values:\n", df.nunique())

# Dropping bins, if any
bin_columns = ["Income_bin","CoapplicantIncome_bin","LoanAmount_bin","TotalIncome","TotalIncome_bin"]
def drop_bins(df, bin_columns):
    df = df.drop(columns=bin_columns)
    return df

# removing 3+ in Dependents and converting Loan_Status to 0 and 1
def clean_categorical_data(train, test):
    train['Dependents'].replace('3+',3,inplace=True)
    test['Dependents'].replace('3+',3,inplace=True)
    train['Loan_Status'].replace('N', 0,inplace=True)
    train['Loan_Status'].replace('Y', 1,inplace=True)
    return train, test

# Correlation Heatmap
def correlation_heatmap(train):
    matrix = train.select_dtypes(include=['number']).corr()
    #matrix = train.corr()
    f, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu",annot=True);
    plt.show()


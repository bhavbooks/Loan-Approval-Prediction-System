import pandas as pd

train = pd.read_csv("data/train_preprocessed.csv")
print(train.shape)
print(train.head())
X_train = train.drop("Loan_Status", axis=1)
y_train = train["Loan_Status"]
print(X_train.shape)
print(y_train.shape)
print(X_train.columns)
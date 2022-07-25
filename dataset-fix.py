import csv

import pandas as pd

names = ("target", "ids", "date", "flag", "user", "text")
train_df = pd.read_csv(
    "training.1600000.processed.noemoticon-ja.csv", names=names)

print(train_df.describe())
print("duplicated", train_df.duplicated(
    subset=["target", "ids"], keep=False).value_counts())
train_df = train_df.drop_duplicates(
    subset=["ids"], keep=False)
print("target: 0", (train_df["target"] == 0).sum())
print("target: 4", (train_df["target"] == 4).sum())

train_df = train_df.dropna()
print(train_df["target"].value_counts())

train_df.to_csv("training.1600000.processed.noemoticon-ja-fixed.csv",
                index=False, quoting=csv.QUOTE_NONNUMERIC)

import pandas as pd

df = pd.read_csv("selected_metadata.csv")
print("Unique captions:", df["text"].nunique())
print(df["text"].value_counts())
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_dataset():
    data_path = Path("data/train.csv")
    if not data_path.exists():
        raise FileNotFoundError("data/train.csv not found")

    df = pd.read_csv(data_path)
    print(f"Total samples in train.csv: {len(df)}")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_split_path = Path("data/train_split.csv")
    val_split_path = Path("data/val_split.csv")

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)

    print(f"Saved {len(train_df)} training samples to {train_split_path}")
    print(f"Saved {len(val_df)} validation samples to {val_split_path}")

    # Check class distributions
    print("\nTraining set distribution per label:")
    print(train_df["label"].value_counts().sort_index())
    print("\nValidation set distribution per label:")
    print(val_df["label"].value_counts().sort_index())

if __name__ == "__main__":
    split_dataset()

import pandas as pd
from pathlib import Path

def check_leakage():
    train_path = Path("data/train.csv")
    test_path = Path("data/unseen_test/test.csv")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Dataset CSV files missing.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_vids = set(train_df["video_id"].astype(str))
    test_vids = set(test_df["video_id"].astype(str))

    exact_overlap = len(train_vids.intersection(test_vids))

    # Base YouTube ID extraction (split by last underscore before timestamp suffix)
    def extract_base_id(vid):
        # Format is typically <youtube_id>_<start_time> (e.g. 6jiO0tPLK7U_000090)
        parts = vid.rsplit("_", 1)
        return parts[0] if len(parts) > 1 else vid

    train_base_ids = set(train_df["video_id"].apply(extract_base_id))
    test_base_ids = set(test_df["video_id"].apply(extract_base_id))

    base_overlap = len(train_base_ids.intersection(test_base_ids))

    print("========================================")
    print("FINAL TEST LEAKAGE CHECK")
    print("========================================")
    print(f"Exact video_id overlap: {exact_overlap}")
    print(f"Base YouTube ID overlap: {base_overlap}")
    print("========================================")

    if exact_overlap > 0 or base_overlap > 0:
        print("CRITICAL ERROR: Data leakage detected between development and test datasets!")
        return False
    else:
        print("VERIFICATION SUCCESSFUL: Zero data leakage between dev and unseen test sets.\n")
        return True

if __name__ == "__main__":
    success = check_leakage()
    if not success:
        exit(1)

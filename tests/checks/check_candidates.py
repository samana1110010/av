import pandas as pd

df_meta = pd.read_csv("data/metadata.csv")
df_train = pd.read_csv("data/train.csv")

# Extract youtube ids
def get_yt(x):
    # e.g., video/6jiO0tPLK7U_000090/video.mp4 or just 6jiO0tPLK7U_000090
    if "/" in x:
        return x.split("/")[-2]
    return x

df_meta['yt_id'] = df_meta['file_name'].apply(get_yt)
df_train['yt_id'] = df_train['video_id']

train_ids = set(df_train['yt_id'].unique())
print(f"Training IDs: {len(train_ids)}")

df_meta_unseen = df_meta[~df_meta['yt_id'].isin(train_ids)]

classes = df_train['class_name'].unique()
df_meta_unseen = df_meta_unseen[df_meta_unseen['text'].isin(classes)]

counts = df_meta_unseen['text'].value_counts()
print("\nUnseen candidates per class:")
print(counts)

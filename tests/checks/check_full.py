import pandas as pd

df_full = pd.read_csv("data/vggsound.csv", header=None, names=['yt_id', 'start_sec', 'class_name', 'split'])
df_train = pd.read_csv("data/train.csv")

# Extract youtube ids from train
def get_yt(x):
    if "/" in x:
        return x.split("/")[-2]
    return x
df_train['yt_id'] = df_train['video_id'].apply(get_yt)
train_ids = set(df_train['yt_id'].unique())

classes = df_train['class_name'].unique()
df_full_filtered = df_full[df_full['class_name'].isin(classes)]

df_unseen = df_full_filtered[~df_full_filtered['yt_id'].isin(train_ids)]

print("Total unseen candidates by class:")
print(df_unseen['class_name'].value_counts())

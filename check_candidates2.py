import pandas as pd

df_meta = pd.read_csv("data/metadata.csv")
df_train = pd.read_csv("data/train.csv")

classes = df_train['class_name'].unique()
print("Classes we want:", classes)

counts = df_meta[df_meta['text'].isin(classes)]['text'].value_counts()
print("\nAll candidates in metadata.csv per class:")
print(counts)


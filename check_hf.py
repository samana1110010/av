from datasets import load_dataset
ds = load_dataset("11hu83/vggsound", split="train")
print("Total rows:", len(ds))
import pandas as pd
df = ds.to_pandas()
classes = [
    'basketball bounce', 'car passing by', 'raining', 'typing on computer keyboard',
    'ripping paper', 'dog barking', 'fire crackling', 'footsteps on snow',
    'door slamming', 'hammering nails'
]
counts = df[df['text'].isin(classes)]['text'].value_counts()
print(counts)

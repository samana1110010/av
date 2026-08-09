import pandas as pd

# Use the metadata downloaded with the project. Importing Hugging Face's
# `datasets` package here collides with this project's local `datasets` package.
df = pd.read_csv("data/metadata.csv")
print("Total rows:", len(df))
classes = [
    'basketball bounce', 'car passing by', 'raining', 'typing on computer keyboard',
    'ripping paper', 'dog barking', 'fire crackling', 'footsteps on snow',
    'door slamming', 'hammering nails'
]
counts = df[df['text'].isin(classes)]['text'].value_counts()
print(counts)

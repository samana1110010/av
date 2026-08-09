import faiss
import numpy as np

# -----------------------
# Load audio embeddings
# -----------------------

embeddings = np.load("embeddings/audio_embeddings.npy")

print("Loaded embeddings:", embeddings.shape)

# -----------------------
# Build FAISS index
# -----------------------

dimension = embeddings.shape[1]

# Embeddings are L2-normalized
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Indexed vectors:", index.ntotal)

# -----------------------
# Save index
# -----------------------

faiss.write_index(index, "embeddings/audio.index")

print("Index saved!")
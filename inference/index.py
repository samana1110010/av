import faiss
import numpy as np

# -----------------------
# Load embeddings
# -----------------------

embeddings = np.load("data/video_embeddings.npy")

print("Loaded embeddings:", embeddings.shape)

# -----------------------
# Build FAISS index
# -----------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Indexed vectors:", index.ntotal)

# -----------------------
# Save index
# -----------------------

faiss.write_index(index, "data/video.index")

print("Index saved!")
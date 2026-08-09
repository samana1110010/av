import faiss
import numpy as np

# -----------------------
# Load index
# -----------------------

index = faiss.read_index("data/video.index")

# -----------------------
# Load embeddings and IDs
# -----------------------

embeddings = np.load("data/video_embeddings.npy")
video_ids = np.load("data/video_ids.npy")

print("Loaded", len(video_ids), "videos")

# -----------------------
# Choose a query
# -----------------------

query_index = 0

query_embedding = embeddings[query_index].reshape(1, -1)

# -----------------------
# Search
# -----------------------

k = 5

distances, indices = index.search(query_embedding, k)

print("\nQuery Video:")
print(video_ids[query_index])

print("\nTop", k, "Nearest Videos:\n")

for rank, idx in enumerate(indices[0], start=1):
    print(
        f"{rank}. {video_ids[idx]} "
        f"(distance={distances[0][rank-1]:.4f})"
    )
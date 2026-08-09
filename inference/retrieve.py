import faiss
import numpy as np

# -----------------------
# Load FAISS audio index
# -----------------------

index = faiss.read_index("embeddings/audio.index")

# -----------------------
# Load embeddings and IDs
# -----------------------

video_embeddings = np.load("data/video_embeddings.npy")
video_ids = np.load("data/video_ids.npy")

audio_ids = np.load("embeddings/audio_ids.npy")

print("Loaded", len(video_ids), "video embeddings")
print("Loaded", len(audio_ids), "audio embeddings")

# -----------------------
# Choose a query video
# -----------------------

query_index = 0

query_embedding = video_embeddings[query_index].reshape(1, -1)

# -----------------------
# Search
# -----------------------

k = 5

scores, indices = index.search(query_embedding, k)

print("\n===================================")
print("Query Video:")
print(video_ids[query_index])
print("===================================\n")

print(f"Top {k} Retrieved Audio Clips:\n")

for rank, idx in enumerate(indices[0], start=1):
    print(
        f"{rank}. {audio_ids[idx]} "
        f"(similarity={scores[0][rank-1]:.4f})"
    )
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load embeddings
# -------------------------

video_embeddings = np.load("embeddings/video_embeddings.npy")
audio_embeddings = np.load("embeddings/audio_embeddings.npy")

print("Video embeddings:", video_embeddings.shape)
print("Audio embeddings:", audio_embeddings.shape)

# -------------------------
# Similarity Matrix
# -------------------------

similarity = cosine_similarity(
    audio_embeddings,
    video_embeddings
)

print("Similarity matrix:", similarity.shape)

# -------------------------
# Recall calculation
# -------------------------

recall1 = 0
recall5 = 0
recall10 = 0

N = similarity.shape[0]

for i in range(N):

    # Highest similarity first
    ranking = np.argsort(similarity[i])[::-1]

    if i in ranking[:1]:
        recall1 += 1

    if i in ranking[:5]:
        recall5 += 1

    if i in ranking[:10]:
        recall10 += 1

recall1 /= N
recall5 /= N
recall10 /= N

print()
print(f"Recall@1  : {recall1:.4f}")
print(f"Recall@5  : {recall5:.4f}")
print(f"Recall@10 : {recall10:.4f}")

# -------------------------
# Show some retrieval examples
# -------------------------

print("\nExample Retrievals\n")

for i in range(5):

    ranking = np.argsort(similarity[i])[::-1]

    print(f"Query Audio #{i}")
    print(f"Correct Video : {i}")
    print(f"Top-5 Retrieved: {ranking[:5]}")
    print()

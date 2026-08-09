from pathlib import Path

import faiss
import numpy as np
import torch

from inference.retrieve import load_audio_index, retrieve_audio
from inference.video_query import load_video_tensor


HAMMER_VIDEO = Path(
    "data/vggsound_selected/video/iYfgk0wAc2E_000001/video.mp4"
)


def test_video_query_preprocessing_is_deterministic():
    first = load_video_tensor(HAMMER_VIDEO)
    second = load_video_tensor(HAMMER_VIDEO)

    assert first.shape == (1, 8, 3, 224, 224)
    assert torch.equal(first, second)


def test_faiss_index_is_built_from_current_embeddings(tmp_path):
    embeddings = np.eye(3, dtype="float32")
    embeddings_path = tmp_path / "audio_embeddings.npy"
    index_path = tmp_path / "audio.index"
    np.save(embeddings_path, embeddings)

    index = load_audio_index(index_path, embeddings_path)

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == 3
    assert index_path.is_file()


def test_arbitrary_hammer_video_retrieves_hammering_audio():
    matches = retrieve_audio(HAMMER_VIDEO, top_k=5)

    assert len(matches) == 5
    assert matches[0]["class_name"] == "hammering nails"
    assert matches[0]["score"] > 0.5

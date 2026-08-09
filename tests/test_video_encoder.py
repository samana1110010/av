import torch

from models.video_encoder import VideoEncoder


def test_video_encoder_output_is_normalized():
    model = VideoEncoder(weights=None).eval()
    frames = torch.rand(1, 1, 3, 64, 64)

    with torch.no_grad():
        embedding = model(frames)

    assert embedding.shape == (1, 128)
    assert torch.allclose(embedding.norm(dim=1), torch.ones(1), atol=1e-5)

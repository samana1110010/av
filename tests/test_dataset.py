import torch

from datasets.video_dataset import VideoDataset


def test_video_dataset_loads_eight_frames():
    dataset = VideoDataset("data/train.csv")
    frames, label, video_id = dataset[0]

    assert len(dataset) > 0
    assert frames.shape == (8, 3, 224, 224)
    assert label.dtype == torch.long
    assert isinstance(video_id, str)


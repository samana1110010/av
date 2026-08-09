import torch

from datasets.audio_dataset import AudioDataset


def test_audio_dataset_has_fixed_size_mel_spectrogram():
    dataset = AudioDataset("data/train.csv", "data/audio")
    mel, label = dataset[0]

    assert mel.shape == (1, 128, 800)
    assert mel.dtype == torch.float32
    assert label.dtype == torch.long

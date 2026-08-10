import torch
from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.loss import SupervisedInfoNCELoss

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ve = VideoEncoder(128).to(device)
    ae = AudioEncoder(128).to(device)
    loss_fn = SupervisedInfoNCELoss(0.07).to(device)
    
    frames = torch.randn(8, 8, 3, 224, 224).to(device)
    mel = torch.randn(8, 1, 128, 800).to(device)
    labels = torch.randint(0, 10, (8,)).to(device)
    
    v_emb = ve(frames)
    a_emb = ae(mel)
    
    loss = loss_fn(v_emb, a_emb, labels)
    loss.backward()
    
    print(f"Forward & Backward successful. Loss: {loss.item()}")
    if torch.cuda.is_available():
        print(f"Max VRAM allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

if __name__ == "__main__":
    test()

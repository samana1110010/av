import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.utils.data import DataLoader

from datasets.multimodal_dataset import MultimodalDataset
from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.loss import SupervisedInfoNCELoss


# -------------------------
# Configuration
# -------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 55
LEARNING_RATE = 5e-5

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# -------------------------
# Dataset
# -------------------------

dataset = MultimodalDataset(
    csv_file="data/train.csv",
    audio_dir="data/audio",
    augment=True,
)

print(f"Samples: {len(dataset)}")


# -------------------------
# DataLoader
# -------------------------

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)


# -------------------------
# Models
# -------------------------

video_encoder = VideoEncoder(
    embedding_dim=128
).to(DEVICE)

audio_encoder = AudioEncoder(
    embedding_dim=128
).to(DEVICE)


# -------------------------
# Loss
# -------------------------

loss_fn = SupervisedInfoNCELoss(
    temperature=0.07
)


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    list(video_encoder.parameters()) +
    list(audio_encoder.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# -------------------------
# Checkpoint directory
# -------------------------

os.makedirs("checkpoints", exist_ok=True)


# -------------------------
# Evaluation Function
# -------------------------

def evaluate(video_encoder, audio_encoder, loader, device):
    video_encoder.eval()
    audio_encoder.eval()
    
    video_embeddings = []
    audio_embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for frames, mel, label, _ in loader:
            frames = frames.to(device)
            mel = mel.to(device)
            video_embeddings.append(video_encoder(frames).cpu())
            audio_embeddings.append(audio_encoder(mel).cpu())
            labels_list.extend(label.numpy())
            
    video_embeddings = torch.cat(video_embeddings).numpy()
    audio_embeddings = torch.cat(audio_embeddings).numpy()
    
    similarity = cosine_similarity(audio_embeddings, video_embeddings)
    
    top1 = 0
    N = len(labels_list)
    for i in range(N):
        ranking = np.argsort(similarity[i])[::-1]
        if labels_list[i] == labels_list[ranking[0]]:
            top1 += 1
            
    return 100 * top1 / N


# -------------------------
# Training
# -------------------------

best_top1 = 0.0
patience_counter = 0
PATIENCE = 3 # Stop if no improvement for 3 evaluations (15 epochs)

for epoch in range(EPOCHS):

    video_encoder.train()
    audio_encoder.train()

    total_loss = 0.0

    for frames, mel, labels, video_ids in loader:

        frames = frames.to(DEVICE, non_blocking=True)
        mel = mel.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        video_embeddings = video_encoder(frames)
        audio_embeddings = audio_encoder(mel)

        loss = loss_fn(
            video_embeddings,
            audio_embeddings,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    # Save checkpoint every epoch
    torch.save(
        video_encoder.state_dict(),
        f"checkpoints/video_encoder_epoch{epoch+1}.pt"
    )

    torch.save(
        audio_encoder.state_dict(),
        f"checkpoints/audio_encoder_epoch{epoch+1}.pt"
    )

    # Evaluation every 5 epochs
    if (epoch + 1) % 5 == 0:
        top1_acc = evaluate(video_encoder, audio_encoder, loader, DEVICE)
        print(f"--> Evaluation Top-1 Accuracy: {top1_acc:.2f}%")
        
        if top1_acc > best_top1:
            best_top1 = top1_acc
            patience_counter = 0
            print("--> New best model!")
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered!")
            break


print("\nTraining complete!")

torch.save(
    video_encoder.state_dict(),
    "checkpoints/video_encoder_final.pt"
)

torch.save(
    audio_encoder.state_dict(),
    "checkpoints/audio_encoder_final.pt"
)

print("Final models saved.")

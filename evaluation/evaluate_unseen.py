import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from datasets.multimodal_dataset import MultimodalDataset

def evaluate_unseen():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Models...")
    video_encoder = VideoEncoder(embedding_dim=128).to(DEVICE)
    audio_encoder = AudioEncoder(embedding_dim=128).to(DEVICE)
    
    video_encoder.load_state_dict(torch.load("checkpoints/video_encoder_final.pt", map_location=DEVICE, weights_only=True))
    audio_encoder.load_state_dict(torch.load("checkpoints/audio_encoder_final.pt", map_location=DEVICE, weights_only=True))
    
    video_encoder.eval()
    audio_encoder.eval()
    
    print("Processing Existing Training Gallery (Audio)...")
    train_dataset = MultimodalDataset("data/train.csv", "data/audio")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    
    gallery_audio_embs = []
    gallery_labels = []
    
    with torch.no_grad():
        for _, mel, label, _ in train_loader:
            mel = mel.to(DEVICE)
            emb = audio_encoder(mel).cpu().numpy()
            gallery_audio_embs.append(emb)
            gallery_labels.extend(label.numpy())
            
    gallery_audio_embs = np.concatenate(gallery_audio_embs)
    gallery_labels = np.array(gallery_labels)
    
    print("Processing Unseen Test Queries (Video)...")
    test_dataset = MultimodalDataset("data/unseen_test/test.csv", "data/unseen_test/audio")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Mapping for labels
    df_train = pd.read_csv("data/train.csv")
    label_to_class = df_train[['label', 'class_name']].drop_duplicates().set_index('label')['class_name'].to_dict()
    
    top1_correct = 0
    top5_correct = 0
    total = len(test_dataset)
    class_stats = {label: {"top1": 0, "top5": 0, "total": 0} for label in label_to_class.keys()}
    
    print("\n---------------------------------------------------------")
    print("UNSEEN VIDEO -> EXISTING AUDIO SEMANTIC RETRIEVAL RESULTS")
    print("---------------------------------------------------------")
    
    with torch.no_grad():
        for i, (frames, _, label, vid_id) in enumerate(test_loader):
            frames = frames.to(DEVICE)
            q_emb = video_encoder(frames).cpu().numpy() # [1, 128]
            
            sim = cosine_similarity(q_emb, gallery_audio_embs)[0]
            ranking = np.argsort(sim)[::-1]
            
            true_label = int(label[0].numpy())
            
            top1_retrieved = gallery_labels[ranking[0]]
            top5_retrieved = gallery_labels[ranking[:5]]
            
            # Semantic matching
            is_top1 = (true_label == top1_retrieved)
            is_top5 = (true_label in top5_retrieved)
            
            if is_top1:
                top1_correct += 1
                class_stats[true_label]["top1"] += 1
            if is_top5:
                top5_correct += 1
                class_stats[true_label]["top5"] += 1
                
            class_stats[true_label]["total"] += 1
            
            # Print detailed per-sample result
            print(f"\nQuery Video ID: {vid_id[0]}")
            print(f"True Class: {label_to_class[true_label]}")
            print(f"Top-1 Retrieved: {label_to_class[top1_retrieved]} (Sim: {sim[ranking[0]]:.3f})")
            top5_classes = [label_to_class[L] for L in top5_retrieved]
            print(f"Top-5 Retrieved: {', '.join(top5_classes)}")
    
    # Calculate overall
    overall_top1 = 100 * top1_correct / total
    overall_top5 = 100 * top5_correct / total
    
    print("\n---------------------------------------------------------")
    print("OVERALL UNSEEN ACCURACY")
    print("---------------------------------------------------------")
    print(f"Top-1 Semantic Accuracy: {overall_top1:.2f}%")
    print(f"Top-5 Semantic Accuracy: {overall_top5:.2f}%")
    
    print("\nPER-CLASS ACCURACY")
    print("---------------------------------------------------------")
    for label, stats in class_stats.items():
        c_name = label_to_class[label]
        c_tot = stats["total"]
        if c_tot > 0:
            c_top1 = 100 * stats["top1"] / c_tot
            print(f"{c_name:<25}: {c_top1:6.2f}%")
        else:
            print(f"{c_name:<25}: No samples found.")
        
if __name__ == "__main__":
    evaluate_unseen()

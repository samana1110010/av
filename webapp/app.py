import os
import sys
import torch
import numpy as np
import torchaudio
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from datasets.multimodal_dataset import MultimodalDataset

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading models...")
video_encoder = VideoEncoder(embedding_dim=128).to(DEVICE)
audio_encoder = AudioEncoder(embedding_dim=128).to(DEVICE)

# Try to load final checkpoints
try:
    video_encoder.load_state_dict(torch.load("checkpoints/video_encoder_final.pt", map_location=DEVICE, weights_only=True))
    audio_encoder.load_state_dict(torch.load("checkpoints/audio_encoder_final.pt", map_location=DEVICE, weights_only=True))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load final checkpoints: {e}")

video_encoder.eval()
audio_encoder.eval()

# Precompute dataset gallery
print("Precomputing dataset gallery...")
dataset = MultimodalDataset("data/train.csv", "data/audio")
# Note: we need the raw metadata (paths) for the frontend to display them
df = pd.read_csv("data/train.csv")

gallery_video_embs = []
gallery_audio_embs = []
gallery_metadata = []

with torch.no_grad():
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    for frames, mel, label, vid_id in loader:
        v_emb = video_encoder(frames.to(DEVICE)).cpu().numpy()
        a_emb = audio_encoder(mel.to(DEVICE)).cpu().numpy()
        gallery_video_embs.append(v_emb)
        gallery_audio_embs.append(a_emb)

gallery_video_embs = np.concatenate(gallery_video_embs)
gallery_audio_embs = np.concatenate(gallery_audio_embs)
print("Gallery ready.")

def process_video_query(file_path):
    # This is a mock function for the uploaded video query
    # In a real scenario, you'd extract frames with OpenCV/ffmpeg
    # For now, we'll pick a random video from the dataset if they just upload a dummy
    pass

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)

@app.route("/data/<path:path>")
def data_files(path):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    return send_from_directory(data_dir, path)

@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    """
    Simulated retrieve endpoint. In a full production app, you'd extract frames 
    or mel spectrograms from the uploaded request.files.
    For this demo frontend, we will pick a random sample from the dataset 
    based on the requested query type to show how the UI works.
    """
    query_type = request.form.get("type", "video") # 'video' or 'audio'
    
    # Pick a random dataset index as the "query"
    idx = np.random.randint(0, len(dataset))
    
    if query_type == "video":
        query_emb = gallery_video_embs[idx:idx+1]
        target_embs = gallery_audio_embs
    else:
        query_emb = gallery_audio_embs[idx:idx+1]
        target_embs = gallery_video_embs
        
    sim = cosine_similarity(query_emb, target_embs)[0]
    
    # Get top 5 matches
    ranking = np.argsort(sim)[::-1][:5]
    
    results = []
    for r in ranking:
        row = df.iloc[r]
        res_type = "audio" if query_type == "video" else "video"
        results.append({
            "id": str(row["video_id"]),
            "title": f"{str(row['class_name']).title()}",
            "score": round(float(sim[r]) * 100, 1),
            "type": res_type,
            "audio_url": f"/data/audio/{row['video_id']}.wav",
            "frames": [f"/data/frames/{row['video_id']}/{i}.jpg" for i in range(8)]
        })
        
    query_info = {
        "id": str(df.iloc[idx]["video_id"]),
        "title": f"Uploaded {query_type.capitalize()} (Class {df.iloc[idx]['label']})"
    }
    
    return jsonify({
        "success": True,
        "query": query_info,
        "results": results
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

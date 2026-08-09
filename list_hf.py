from huggingface_hub import list_repo_files
import json

files = list_repo_files(repo_id="11hu83/vggsound", repo_type="dataset")
videos = [f for f in files if f.endswith('.mp4')]
print(f"Total videos in repo: {len(videos)}")

with open('hf_videos.json', 'w') as f:
    json.dump(videos, f)

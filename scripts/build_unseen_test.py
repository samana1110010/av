import pandas as pd
import subprocess
import os
import shutil

df_full = pd.read_csv("data/vggsound.csv", header=None, names=['yt_id', 'start_sec', 'class_name', 'split'])
df_train = pd.read_csv("data/train.csv")

def get_yt(x):
    if "/" in x:
        x = x.split("/")[-2]
    return x.rsplit("_", 1)[0]

df_train['yt_id'] = df_train['video_id'].apply(get_yt)
train_ids = set(df_train['yt_id'].unique())

classes = df_train['class_name'].unique()
df_full_filtered = df_full[df_full['class_name'].isin(classes)]

df_unseen = df_full_filtered[~df_full_filtered['yt_id'].isin(train_ids)]

os.makedirs("data/unseen_test/raw/video", exist_ok=True)

class_map = df_train[['class_name', 'label']].drop_duplicates().set_index('class_name')['label'].to_dict()

downloaded_data = []

# Keep trying until we get 5 successful downloads per class
for class_name in classes:
    class_candidates = df_unseen[df_unseen['class_name'] == class_name].sample(frac=1, random_state=42)
    downloaded_count = 0
    print(f"\nProcessing class: {class_name}")
    
    for idx, row in class_candidates.iterrows():
        if downloaded_count >= 5:
            break
            
        yt_id = row['yt_id']
        start_sec = int(row['start_sec'])
        video_id = f"{yt_id}_{start_sec:06d}"
        
        print(f"  Attempting to download {video_id}...")
        
        vid_dir = f"data/unseen_test/raw/video/{video_id}"
        os.makedirs(vid_dir, exist_ok=True)
        vid_path = f"{vid_dir}/video.mp4"
        
        # Download exactly 10 seconds starting from start_sec
        dl_cmd = [
            "uv", "run", "yt-dlp",
            f"https://www.youtube.com/watch?v={yt_id}",
            "-f", "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
            "--download-sections", f"*{start_sec}-{start_sec + 10}",
            "--force-keyframes-at-cuts",
            "-o", vid_path,
            "--max-downloads", "1"
        ]
        
        try:
            res = subprocess.run(dl_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=60)
            if res.returncode in [0, 101] and os.path.exists(vid_path):
                print(f"  -> Successfully downloaded {video_id}")
                downloaded_data.append({
                    "video_id": video_id,
                    "class_name": class_name,
                    "label": class_map[class_name],
                    "frame_folder": f"data/unseen_test/frames/{video_id}"
                })
                downloaded_count += 1
            else:
                print(f"  -> Failed or video unavailable.")
                shutil.rmtree(vid_dir)
        except subprocess.TimeoutExpired:
            print(f"  -> Timed out.")
            shutil.rmtree(vid_dir, ignore_errors=True)

out_df = pd.DataFrame(downloaded_data)
out_df.to_csv("data/unseen_test/test.csv", index=False)
print("\nDone downloading evaluation set!")

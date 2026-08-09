from huggingface_hub import list_repo_files
files = list_repo_files(repo_id="11hu83/vggsound", repo_type="dataset")
videos = [f for f in files if f.endswith('.mp4')]
print("Total videos in HuggingFace repo:", len(videos))

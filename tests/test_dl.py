import subprocess
dl_cmd = [
    "uv", "run", "yt-dlp",
    f"https://www.youtube.com/watch?v=KEeyw8lcPKs",
    "-f", "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
    "--download-sections", f"*0-10",
    "--force-keyframes-at-cuts",
    "-o", "data/unseen_test/raw/video/KEeyw8lcPKs_000000/video.mp4",
    "--max-downloads", "1"
]
print("Running command")
res = subprocess.run(dl_cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)

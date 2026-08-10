from inference.sync_video import sync_video

video = "data/vggsound_selected/video/6jiO0tPLK7U_000090/video.mp4"
audio = "data/audio/6jiO0tPLK7U_000090.wav"

sync_video(video, audio, "output.mp4")

print("Done!")
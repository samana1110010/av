from io import BytesIO
from pathlib import Path
import json
import subprocess

from webapp.app import GENERATED_DIR, app


def test_health_and_home_are_ready():
    client = app.test_client()

    health = client.get("/api/health")
    home = client.get("/")

    assert health.status_code == 200
    assert health.get_json()["gallery_size"] == 500
    assert health.get_json()["version"] == "2026.08.09-5"
    assert b"Upload a silent video" in home.data
    assert b"style.css?v=20260809-3" in home.data
    assert home.headers["Cache-Control"] == "no-store, max-age=0"


def test_static_assets_are_linked_and_not_cached():
    client = app.test_client()
    css = client.get("/style.css?v=20260809-3")
    javascript = client.get("/script.js?v=20260809-3")

    assert css.status_code == 200
    assert b"--acid" in css.data
    assert css.headers["Cache-Control"] == "no-store, max-age=0"
    assert javascript.status_code == 200
    assert b"checkHealth" in javascript.data


def test_retrieve_requires_a_file():
    response = app.test_client().post("/api/retrieve", data={"type": "video"})

    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_retrieve_rejects_wrong_extension():
    response = app.test_client().post(
        "/api/retrieve",
        data={"type": "audio", "file": (BytesIO(b"not audio"), "query.txt")},
    )

    assert response.status_code == 415


def test_real_audio_query_returns_ranked_video_results():
    audio_path = Path("data/audio/6jiO0tPLK7U_000090.wav")
    with audio_path.open("rb") as audio:
        response = app.test_client().post(
            "/api/retrieve",
            data={"type": "audio", "file": (audio, audio_path.name)},
        )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert len(payload["results"]) == 5
    assert [result["rank"] for result in payload["results"]] == [1, 2, 3, 4, 5]
    assert all(result["type"] == "video" for result in payload["results"])


def test_video_query_creates_playable_downloadable_output():
    video_path = Path(
        "data/vggsound_selected/video/6jiO0tPLK7U_000090/video.mp4"
    )
    with video_path.open("rb") as video:
        response = app.test_client().post(
            "/api/retrieve",
            data={"type": "video", "file": (video, video_path.name)},
        )

    payload = response.get_json()
    output_path = GENERATED_DIR / Path(payload["output"]["url"]).name
    preview = None
    download = None
    try:
        assert response.status_code == 200
        assert payload["output"]["audio"]["id"] == payload["results"][0]["id"]
        preview = app.test_client().get(payload["output"]["url"])
        download = app.test_client().get(payload["output"]["download_url"])
        assert preview.status_code == 200
        assert preview.mimetype == "video/mp4"
        assert b"ftyp" in preview.data[:32]
        assert download.headers["Content-Disposition"].startswith("attachment;")
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,sample_rate", "-of", "json", str(output_path),
        ], text=True))
        assert {stream["codec_type"] for stream in probe["streams"]} == {"video", "audio"}
        audio_stream = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "audio"
        )
        assert audio_stream["sample_rate"] == "48000"
    finally:
        if preview is not None:
            preview.close()
        if download is not None:
            download.close()
        output_path.unlink(missing_ok=True)

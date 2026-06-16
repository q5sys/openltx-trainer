"""Integration tests for the dataset pipeline endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_probe_source(client, tmp_path: Path):
    """Probe endpoint returns source media info from the fake pipeline."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")

    resp = client.post("/api/dataset/probe", json={"source_path": str(source)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test_video.mp4"
    assert data["duration_s"] == 30.0
    assert data["width"] == 1920
    assert data["height"] == 1080
    assert data["is_image"] is False


def test_probe_source_missing_file(client):
    """Probe raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        client.post("/api/dataset/probe", json={"source_path": "/no/such/file.mp4"})


def test_detect_scenes(client, tmp_path: Path):
    """Scene detection returns proposals from the fake pipeline."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")

    resp = client.post("/api/dataset/scenes/detect", json={
        "source_path": str(source),
        "threshold": 27.0,
        "target_clip_length_s": 5.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert data[0]["scene_index"] == 0
    assert data[0]["start_s"] == 0.0
    assert data[0]["end_s"] == 10.0
    assert data[0]["length_status"] == "long"


def test_create_clip(client, tmp_path: Path):
    """Creating a clip writes files and returns a ClipResult."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    resp = client.post("/api/dataset/clips", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
        "start_s": 0.0,
        "end_s": 5.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["duration_s"] == 5.0
    assert data["width"] == 1280
    assert data["height"] == 720
    assert data["clip_id"]  # non-empty

    # Verify files were created on disk by the fake.
    clips_dir = dataset_dir / "clips"
    assert clips_dir.exists()
    mp4_files = list(clips_dir.glob("*.mp4"))
    assert len(mp4_files) == 1


def test_create_clips_batch(client, tmp_path: Path):
    """Batch clip creation creates multiple clips."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    resp = client.post("/api/dataset/clips/batch", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
        "segments": [
            {"start_s": 0.0, "end_s": 5.0},
            {"start_s": 5.0, "end_s": 10.0},
            {"start_s": 10.0, "end_s": 15.0},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3

    clips_dir = dataset_dir / "clips"
    mp4_files = list(clips_dir.glob("*.mp4"))
    assert len(mp4_files) == 3


def test_list_clips(client, tmp_path: Path):
    """List clips returns clips from the dataset directory."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create a clip first.
    client.post("/api/dataset/clips", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
        "start_s": 0.0,
        "end_s": 5.0,
    })

    resp = client.post("/api/dataset/clips/list", json={
        "dataset_dir": str(dataset_dir),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["filename"].endswith(".mp4")


def test_delete_clip(client, tmp_path: Path):
    """Deleting a clip removes its files."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create a clip.
    create_resp = client.post("/api/dataset/clips", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
        "start_s": 0.0,
        "end_s": 5.0,
    })
    clip_id = create_resp.json()["clip_id"]

    # Delete it.
    resp = client.post("/api/dataset/clips/delete", json={
        "dataset_dir": str(dataset_dir),
        "clip_id": clip_id,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify it is gone.
    list_resp = client.post("/api/dataset/clips/list", json={
        "dataset_dir": str(dataset_dir),
    })
    assert len(list_resp.json()) == 0


def test_update_caption(client, tmp_path: Path):
    """Caption update writes the text to the clip's .txt sidecar."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    create_resp = client.post("/api/dataset/clips", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
        "start_s": 0.0,
        "end_s": 5.0,
    })
    clip_id = create_resp.json()["clip_id"]

    resp = client.post("/api/dataset/clips/caption", json={
        "dataset_dir": str(dataset_dir),
        "clip_id": clip_id,
        "caption": "A person walking in a park",
    })
    assert resp.status_code == 200

    # Verify the caption file was written.
    caption_path = dataset_dir / "clips" / f"{clip_id}.txt"
    assert caption_path.read_text() == "A person walking in a park"


def test_get_thumbnail(client, tmp_path: Path):
    """Thumbnail endpoint returns base64 data."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    create_resp = client.post("/api/dataset/clips", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
        "start_s": 0.0,
        "end_s": 5.0,
    })
    clip_id = create_resp.json()["clip_id"]

    resp = client.post("/api/dataset/clips/thumbnail", json={
        "dataset_dir": str(dataset_dir),
        "clip_id": clip_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["clip_id"] == clip_id
    assert len(data["thumbnail_b64"]) > 0


def test_import_image(client, tmp_path: Path):
    """Importing an image creates a PNG entry in images/."""
    source = tmp_path / "test_image.png"
    source.write_bytes(b"fake-image-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    resp = client.post("/api/dataset/images", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["duration_s"] == 0.0
    assert data["filename"].endswith(".png")

    images_dir = dataset_dir / "images"
    assert images_dir.exists()
    png_files = list(images_dir.glob("*.png"))
    assert len(png_files) == 1

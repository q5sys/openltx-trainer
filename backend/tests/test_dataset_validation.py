"""Integration tests for dataset validation and trigger word endpoints."""

from __future__ import annotations

from pathlib import Path


def _create_dataset_with_clips(client, tmp_path: Path, count: int = 3) -> tuple[Path, list[str]]:
    """Helper: create a dataset with N clips, each with a caption."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    clip_ids = []
    for i in range(count):
        resp = client.post("/api/dataset/clips", json={
            "source_path": str(source),
            "dataset_dir": str(dataset_dir),
            "start_s": i * 5.0,
            "end_s": (i + 1) * 5.0,
        })
        clip_id = resp.json()["clip_id"]
        clip_ids.append(clip_id)

        # Write a caption for each clip.
        client.post("/api/dataset/clips/caption", json={
            "dataset_dir": str(dataset_dir),
            "clip_id": clip_id,
            "caption": f"ohwx A person walking in scene {i}, doing something interesting and unique.",
        })

    return dataset_dir, clip_ids


# ---------------------------------------------------------------
# Trigger word validation
# ---------------------------------------------------------------


def test_validate_trigger_valid(client):
    """A valid trigger word passes validation."""
    resp = client.post("/api/dataset/trigger/validate", json={"trigger": "ohwx"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["error"] is None
    assert data["warning"] is None


def test_validate_trigger_empty(client):
    """Empty trigger is invalid."""
    resp = client.post("/api/dataset/trigger/validate", json={"trigger": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert "empty" in data["error"].lower()


def test_validate_trigger_too_short(client):
    """Single character trigger is invalid."""
    resp = client.post("/api/dataset/trigger/validate", json={"trigger": "x"})
    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_validate_trigger_special_chars(client):
    """Trigger with special characters is invalid."""
    resp = client.post("/api/dataset/trigger/validate", json={"trigger": "oh-wx!"})
    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_validate_trigger_stopword(client):
    """Stoplist word is valid but has a warning."""
    resp = client.post("/api/dataset/trigger/validate", json={"trigger": "person"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["warning"] is not None
    assert "common word" in data["warning"].lower()


# ---------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------


def test_validate_dataset_valid(client, tmp_path: Path):
    """A well-formed dataset passes validation."""
    dataset_dir, clip_ids = _create_dataset_with_clips(client, tmp_path, count=3)

    resp = client.post("/api/dataset/validate", json={
        "dataset_dir": str(dataset_dir),
        "trigger": "ohwx",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert len(data["errors"]) == 0
    assert data["stats"]["clip_count"] == 3
    assert data["stats"]["captioned"] == 3
    assert data["stats"]["trigger_present"] == 3


def test_validate_dataset_empty(client, tmp_path: Path):
    """Empty dataset fails with NO_CLIPS error."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    resp = client.post("/api/dataset/validate", json={
        "dataset_dir": str(dataset_dir),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    error_codes = [e["code"] for e in data["errors"]]
    assert "NO_CLIPS" in error_codes


def test_validate_dataset_missing_caption(client, tmp_path: Path):
    """Clips without captions produce MISSING_CAPTION errors."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create clip without writing a caption.
    client.post("/api/dataset/clips", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
        "start_s": 0.0,
        "end_s": 5.0,
    })

    resp = client.post("/api/dataset/validate", json={
        "dataset_dir": str(dataset_dir),
    })
    data = resp.json()
    assert data["valid"] is False
    error_codes = [e["code"] for e in data["errors"]]
    assert "MISSING_CAPTION" in error_codes


def test_validate_dataset_missing_trigger(client, tmp_path: Path):
    """Captions without the trigger word produce MISSING_TRIGGER errors."""
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
    clip_id = resp.json()["clip_id"]

    # Write a caption WITHOUT the trigger.
    client.post("/api/dataset/clips/caption", json={
        "dataset_dir": str(dataset_dir),
        "clip_id": clip_id,
        "caption": "A person walking in a park on a sunny day.",
    })

    resp = client.post("/api/dataset/validate", json={
        "dataset_dir": str(dataset_dir),
        "trigger": "ohwx",
    })
    data = resp.json()
    assert data["valid"] is False
    error_codes = [e["code"] for e in data["errors"]]
    assert "MISSING_TRIGGER" in error_codes


def test_validate_dataset_tiny_warning(client, tmp_path: Path):
    """Small dataset produces TINY_DATASET warning."""
    dataset_dir, _ = _create_dataset_with_clips(client, tmp_path, count=3)

    resp = client.post("/api/dataset/validate", json={
        "dataset_dir": str(dataset_dir),
        "trigger": "ohwx",
    })
    data = resp.json()
    warning_codes = [w["code"] for w in data["warnings"]]
    assert "TINY_DATASET" in warning_codes


def test_validate_dataset_stats(client, tmp_path: Path):
    """Stats reflect actual dataset contents."""
    dataset_dir, clip_ids = _create_dataset_with_clips(client, tmp_path, count=5)

    resp = client.post("/api/dataset/validate", json={
        "dataset_dir": str(dataset_dir),
        "trigger": "ohwx",
    })
    stats = resp.json()["stats"]
    assert stats["clip_count"] == 5
    assert stats["captioned"] == 5
    assert stats["trigger_present"] == 5
    assert stats["total_duration_s"] > 0


# ---------------------------------------------------------------
# Trigger audit
# ---------------------------------------------------------------


def test_audit_trigger_all_present(client, tmp_path: Path):
    """Audit returns empty when all captions contain the trigger."""
    dataset_dir, _ = _create_dataset_with_clips(client, tmp_path, count=3)

    resp = client.post("/api/dataset/trigger/audit", json={
        "dataset_dir": str(dataset_dir),
        "trigger": "ohwx",
    })
    assert resp.status_code == 200
    assert resp.json() == []


def test_audit_trigger_some_missing(client, tmp_path: Path):
    """Audit returns clips missing the trigger."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create 2 clips: one with trigger, one without.
    for i, caption in enumerate(["ohwx person walking", "person running without trigger"]):
        resp = client.post("/api/dataset/clips", json={
            "source_path": str(source),
            "dataset_dir": str(dataset_dir),
            "start_s": i * 5.0,
            "end_s": (i + 1) * 5.0,
        })
        clip_id = resp.json()["clip_id"]
        client.post("/api/dataset/clips/caption", json={
            "dataset_dir": str(dataset_dir),
            "clip_id": clip_id,
            "caption": caption,
        })

    resp = client.post("/api/dataset/trigger/audit", json={
        "dataset_dir": str(dataset_dir),
        "trigger": "ohwx",
    })
    data = resp.json()
    assert len(data) == 1
    assert "running" in data[0]["caption"]


# ---------------------------------------------------------------
# Trigger prepend
# ---------------------------------------------------------------


def test_prepend_trigger(client, tmp_path: Path):
    """Prepend adds the trigger to captions that lack it."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create clips: 2 without trigger, 1 with trigger.
    clip_ids = []
    for i, caption in enumerate(["person walking", "person running", "ohwx person standing"]):
        resp = client.post("/api/dataset/clips", json={
            "source_path": str(source),
            "dataset_dir": str(dataset_dir),
            "start_s": i * 5.0,
            "end_s": (i + 1) * 5.0,
        })
        clip_id = resp.json()["clip_id"]
        clip_ids.append(clip_id)
        client.post("/api/dataset/clips/caption", json={
            "dataset_dir": str(dataset_dir),
            "clip_id": clip_id,
            "caption": caption,
        })

    resp = client.post("/api/dataset/trigger/prepend", json={
        "dataset_dir": str(dataset_dir),
        "trigger": "ohwx",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["modified_count"] == 2

    # Verify captions on disk.
    for clip_id in clip_ids:
        caption_path = dataset_dir / "clips" / f"{clip_id}.txt"
        assert caption_path.read_text().startswith("ohwx")


def test_prepend_trigger_specific_clips(client, tmp_path: Path):
    """Prepend with clip_ids only modifies specified clips."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    clip_ids = []
    for i in range(3):
        resp = client.post("/api/dataset/clips", json={
            "source_path": str(source),
            "dataset_dir": str(dataset_dir),
            "start_s": i * 5.0,
            "end_s": (i + 1) * 5.0,
        })
        clip_id = resp.json()["clip_id"]
        clip_ids.append(clip_id)
        client.post("/api/dataset/clips/caption", json={
            "dataset_dir": str(dataset_dir),
            "clip_id": clip_id,
            "caption": f"person in scene {i} doing something interesting and noteworthy.",
        })

    # Only prepend to the first clip.
    resp = client.post("/api/dataset/trigger/prepend", json={
        "dataset_dir": str(dataset_dir),
        "trigger": "ohwx",
        "clip_ids": [clip_ids[0]],
    })
    assert resp.json()["modified_count"] == 1

    # First clip has trigger, second does not.
    first_caption = (dataset_dir / "clips" / f"{clip_ids[0]}.txt").read_text()
    second_caption = (dataset_dir / "clips" / f"{clip_ids[1]}.txt").read_text()
    assert first_caption.startswith("ohwx")
    assert not second_caption.startswith("ohwx")

"""Integration-style tests for checkpoint recommendation and download endpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from _routes._errors import HTTPError
import handlers.models_handler as models_handler_module
from runtime_config.model_download_specs import (
    LTXLocalModelDeprecated,
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    resolve_downloading_dir,
    resolve_model_path,
)
from state.app_state_types import DownloadSessionComplete, DownloadSessionError, DownloadingSession, FileDownloadRunning
from tests.http_error_assertions import assert_http_error


def _current_ltx_spec():
    return get_ltx_model_spec(get_latest_ltx_model_id())


def _cp_path(test_state, cp_id: str) -> Path:
    return resolve_model_path(test_state.config.default_models_dir, cp_id)


class TestRecommendations:
    def test_ltx_recommendation_requires_primary_local_bundle(self, client):
        spec = _current_ltx_spec()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [
                spec.model_cp,
                spec.upscale_cp,
                spec.text_encoder_cp,
            ],
        }

    def test_ltx_recommendation_always_includes_text_encoder(self, client, test_state):
        spec = _current_ltx_spec()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [
                spec.model_cp,
                spec.upscale_cp,
                spec.text_encoder_cp,
            ],
        }

    def test_ltx_recommendation_ok_when_required_bundle_is_downloaded(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ltx_recommendation_reports_missing_text_encoder_for_current_model(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        text_encoder_path = _cp_path(test_state, _current_ltx_spec().text_encoder_cp)
        for child in text_encoder_path.iterdir():
            child.unlink()
        text_encoder_path.rmdir()

        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [_current_ltx_spec().text_encoder_cp],
        }

    def test_text_encoder_recommendation(self, client, create_fake_model_files, test_state):
        create_fake_model_files()
        text_encoder_path = _cp_path(test_state, _current_ltx_spec().text_encoder_cp)
        for child in text_encoder_path.iterdir():
            child.unlink()
        text_encoder_path.rmdir()

        response = client.get("/api/models/text-encoder-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] == _current_ltx_spec().text_encoder_cp
        assert response.json()["expected_size_bytes"] > 0


class TestDownloadProgress:
    def test_unknown_session_returns_404(self, client):
        response = client.get("/api/models/download/progress", params={"sessionId": "nonexistent"})
        assert_http_error(response, status_code=404, code="UNKNOWN_DOWNLOAD_SESSION")

    def test_active_progress(self, client, test_state):
        test_state.state.downloading_session = DownloadingSession(
            id="test-session",
            current_running_file=FileDownloadRunning(
                file_type="ltx-2.3-22b-dev",
                target_path="ltx-2.3-22b-dev.safetensors",
                downloaded_bytes=5_000_000_000,
                speed_bytes_per_sec=50_000_000.0,
            ),
            files_to_download={"ltx-2.3-22b-dev"},
            completed_files=set(),
            completed_bytes=0,
        )
        response = client.get("/api/models/download/progress", params={"sessionId": "test-session"})
        assert response.status_code == 200
        assert response.json()["status"] == "downloading"
        assert response.json()["current_downloading_file"] == "ltx-2.3-22b-dev"

    def test_completed_and_error_sessions(self, client, test_state):
        test_state.state.completed_download_sessions["done-session"] = DownloadSessionComplete()
        test_state.state.completed_download_sessions["err-session"] = DownloadSessionError(error_message="network error")

        complete = client.get("/api/models/download/progress", params={"sessionId": "done-session"})
        assert complete.status_code == 200
        assert complete.json()["status"] == "complete"

        failed = client.get("/api/models/download/progress", params={"sessionId": "err-session"})
        assert failed.status_code == 200
        assert failed.json()["status"] == "error"
        assert failed.json()["error"] == "network error"


class TestModelDownloads:
    def test_download_start_success(self, client, test_state):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": ["gemma-3-12b-it-qat-q4_0-unquantized"]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "started"
        assert _cp_path(test_state, "gemma-3-12b-it-qat-q4_0-unquantized").exists()

    def test_download_conflicts_when_another_session_is_running(self, client, test_state):
        test_state.downloads.start_download({"ltx-2.3-22b-dev"})
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": ["gemma-3-12b-it-qat-q4_0-unquantized"]},
        )
        assert_http_error(response, status_code=409, code="DOWNLOAD_ALREADY_RUNNING")

    def test_upgrade_without_downloaded_model_is_rejected(self, client):
        response = client.post(
            "/api/models/download",
            json={"type": "upgrade", "cp_ids": [_current_ltx_spec().model_cp]},
        )
        assert_http_error(response, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")

    def test_upgrade_raises_500_for_internal_ltx_mapping_inconsistency(self, test_state, monkeypatch):
        monkeypatch.setattr(test_state.models, "_current_downloaded_ltx_model_id", lambda: "ltx-legacy")
        monkeypatch.setattr(models_handler_module, "get_ltx_model_id_for_cp", lambda cp_id: None)

        with pytest.raises(HTTPError) as exc_info:
            test_state.models.resolve_upgrade_download({_current_ltx_spec().model_cp})

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "INVALID_LTX_MODEL_CONFIG"

    def test_upgrade_raises_500_when_latest_ltx_model_is_not_relevant(self, test_state, monkeypatch):
        monkeypatch.setattr(test_state.models, "_current_downloaded_ltx_model_id", lambda: "ltx-legacy")
        monkeypatch.setattr(models_handler_module, "get_latest_ltx_model_id", lambda: "ltx-2.3-22b-dev")
        monkeypatch.setattr(models_handler_module, "get_ltx_model_id_for_cp", lambda cp_id: "ltx-2.3-22b-dev")

        original_get_ltx_model_spec = models_handler_module.get_ltx_model_spec

        def _get_ltx_model_spec(model_id):
            spec = original_get_ltx_model_spec(model_id)
            if model_id == "ltx-2.3-22b-dev":
                return replace(spec, relevance=LTXLocalModelDeprecated())
            return spec

        monkeypatch.setattr(models_handler_module, "get_ltx_model_spec", _get_ltx_model_spec)

        with pytest.raises(HTTPError) as exc_info:
            test_state.models.resolve_upgrade_download({_current_ltx_spec().model_cp})

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "INVALID_LTX_MODEL_CONFIG"

    def test_download_error_is_reported(self, client, test_state):
        test_state.model_downloader.fail_next = RuntimeError("Connection refused")

        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": ["gemma-3-12b-it-qat-q4_0-unquantized"]},
        )
        assert response.status_code == 200
        session_id = response.json()["sessionId"]

        progress = client.get("/api/models/download/progress", params={"sessionId": session_id})
        assert progress.status_code == 200
        assert progress.json()["status"] == "error"

    def test_download_uses_progress_callback(self, client, test_state):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": ["gemma-3-12b-it-qat-q4_0-unquantized"]},
        )
        assert response.status_code == 200
        assert test_state.model_downloader.calls
        assert all(call["on_progress"] is not None for call in test_state.model_downloader.calls)

    def test_failed_download_cleans_staging_dir(self, test_state):
        test_state.model_downloader.fail_next = RuntimeError("network error")
        test_state.downloads.start_model_download(download_type="download", cp_ids={"gemma-3-12b-it-qat-q4_0-unquantized"})
        assert len(test_state.task_runner.errors) == 1
        assert not resolve_downloading_dir(test_state.config.default_models_dir).exists()


class TestCheckpointDeletion:
    def test_delete_missing_checkpoint_is_noop(self, client):
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": ["gemma-3-12b-it-qat-q4_0-unquantized"]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_delete_rejects_current_ltx_bundle(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [_current_ltx_spec().model_cp]},
        )
        assert_http_error(response, status_code=409, code="DELETE_PROTECTED_CHECKPOINT")

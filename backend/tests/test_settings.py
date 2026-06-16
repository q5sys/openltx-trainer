"""Tests for GET /api/settings and POST /api/settings."""

from __future__ import annotations

import json
from pathlib import Path

from state.app_settings import AppSettings, UpdateSettingsRequest
from state import build_initial_state
from app_handler import ServiceBundle
from tests.conftest import TEST_ADMIN_TOKEN
from tests.fakes.services import FakeServices


class TestGetSettings:
    def test_default_settings(self, client, default_app_settings, test_state):
        r = client.get("/api/settings")
        assert r.status_code == 200
        data = r.json()
        assert data["useTorchCompile"] is False
        assert data["keepModelsLoaded"] is True
        assert data["loadOnStartup"] is False
        assert data["defaultGpuIndex"] == 0
        assert data["modelDirs"]["baseModels"] == "auto"
        assert data["trainingDefaults"]["saveOptimizerState"] is True
        assert data["captioningDefaults"]["backend"] == "qwen_vl_local"
        assert data["verificationDefaults"]["defaultCfg"] == 10.0

    def test_api_keys_are_masked(self, client, test_state):
        test_state.state.app_settings.captioning_api_keys.gemini = "sk-test-key-12345678"
        r = client.get("/api/settings")
        data = r.json()
        masked = data["captioningApiKeys"]["gemini"]
        assert masked.endswith("5678")
        assert masked.startswith("*")
        assert "sk-test" not in masked

    def test_empty_api_keys_stay_empty(self, client, test_state):
        r = client.get("/api/settings")
        data = r.json()
        assert data["captioningApiKeys"]["gemini"] == ""

    def test_reflects_changed_settings(self, client, test_state):
        test_state.state.app_settings.use_torch_compile = True
        r = client.get("/api/settings")
        assert r.json()["useTorchCompile"] is True


class TestPostSettings:
    def test_update_single_field(self, client, test_state):
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True

    def test_update_nested_field(self, client, test_state):
        r = client.post("/api/settings", json={
            "trainingDefaults": {"saveOptimizerState": False}
        })
        assert r.status_code == 200
        assert test_state.state.app_settings.training_defaults.save_optimizer_state is False

    def test_update_api_key(self, client, test_state):
        r = client.post("/api/settings", json={
            "captioningApiKeys": {"gemini": "test-key-abc"}
        })
        assert r.status_code == 200
        assert test_state.state.app_settings.captioning_api_keys.gemini == "test-key-abc"

    def test_gpu_index_clamped(self, client, test_state):
        r = client.post("/api/settings", json={"defaultGpuIndex": 99})
        assert r.status_code == 200
        assert test_state.state.app_settings.default_gpu_index <= 15

    def test_verification_cfg_clamped(self, client, test_state):
        r = client.post("/api/settings", json={
            "verificationDefaults": {"defaultCfg": 100.0}
        })
        assert r.status_code == 200
        assert test_state.state.app_settings.verification_defaults.default_cfg <= 30.0

    def test_unknown_field_rejected(self, client):
        r = client.post("/api/settings", json={"unknownSetting": True})
        assert r.status_code == 422


class TestModelDirsAdminGuard:
    def test_model_dirs_requires_admin_token(self, client, test_state):
        r = client.post("/api/settings", json={"modelDirs": {"baseModels": "/tmp/new-models"}})
        assert r.status_code == 403

    def test_model_dirs_with_wrong_admin_token(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={"modelDirs": {"baseModels": "/tmp/new-models"}},
            headers={"X-Admin-Token": "wrong-token"},
        )
        assert r.status_code == 403

    def test_model_dirs_with_valid_admin_token(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={"modelDirs": {"baseModels": "/tmp/new-models"}},
            headers={"X-Admin-Token": TEST_ADMIN_TOKEN},
        )
        assert r.status_code == 200
        assert test_state.state.app_settings.model_dirs.base_models == "/tmp/new-models"

    def test_non_admin_fields_without_admin_token(self, client, test_state):
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True

    def test_effective_models_dir_uses_custom(self, client, test_state):
        test_state.state.app_settings.model_dirs.base_models = "/custom/models"
        assert test_state.models.models_dir == Path("/custom/models")

    def test_effective_models_dir_fallback(self, client, test_state):
        assert test_state.state.app_settings.model_dirs.base_models == "auto"
        assert test_state.models.models_dir == test_state.config.default_models_dir

    def test_model_dirs_persists_and_loads(self, client, test_state, default_app_settings):
        r = client.post(
            "/api/settings",
            json={"modelDirs": {"baseModels": "/tmp/persisted-models"}},
            headers={"X-Admin-Token": TEST_ADMIN_TOKEN},
        )
        assert r.status_code == 200

        fake_services = FakeServices()
        bundle = ServiceBundle(
            http=fake_services.http,
            gpu_cleaner=fake_services.gpu_cleaner,
            model_downloader=fake_services.model_downloader,
            gpu_info=fake_services.gpu_info,
            video_processor=fake_services.video_processor,
            text_encoder=fake_services.text_encoder,
            task_runner=fake_services.task_runner,
            dataset_pipeline=fake_services.dataset_pipeline,
            caption_pipeline=fake_services.caption_pipeline,
            training_supervisor=fake_services.training_supervisor,
            verification_pipeline=fake_services.verification_pipeline,
        )
        loaded = build_initial_state(test_state.config, default_app_settings.model_copy(deep=True), service_bundle=bundle)
        assert loaded.state.app_settings.model_dirs.base_models == "/tmp/persisted-models"
        assert loaded.models.models_dir == Path("/tmp/persisted-models")


class TestSettingsPersistence:
    def _new_state(self, test_state, default_app_settings):
        fake_services = FakeServices()
        bundle = ServiceBundle(
            http=fake_services.http,
            gpu_cleaner=fake_services.gpu_cleaner,
            model_downloader=fake_services.model_downloader,
            gpu_info=fake_services.gpu_info,
            video_processor=fake_services.video_processor,
            text_encoder=fake_services.text_encoder,
            task_runner=fake_services.task_runner,
            dataset_pipeline=fake_services.dataset_pipeline,
            caption_pipeline=fake_services.caption_pipeline,
            training_supervisor=fake_services.training_supervisor,
            verification_pipeline=fake_services.verification_pipeline,
        )
        return build_initial_state(test_state.config, default_app_settings.model_copy(deep=True), service_bundle=bundle)

    def test_load_settings_ignores_deprecated_keys(self, test_state, default_app_settings):
        test_state.config.settings_file.write_text(
            json.dumps(
                {
                    "prompt_cache_size": 5000,
                    "locked_seed": -55,
                    "fast_model": {"use_upscaler": False},
                    "pro_model": {"steps": 999},
                    "use_torch_compile": True,
                }
            ),
            encoding="utf-8",
        )

        loaded = self._new_state(test_state, default_app_settings)
        # Deprecated keys are stripped, but valid keys are preserved
        assert loaded.state.app_settings.use_torch_compile is True
        assert "fast_model" not in loaded.state.app_settings.model_dump(by_alias=False)
        assert "pro_model" not in loaded.state.app_settings.model_dump(by_alias=False)

    def test_torch_compile_persists(self, client, test_state, default_app_settings):
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.use_torch_compile is True

    def test_nested_settings_persist(self, client, test_state, default_app_settings):
        r = client.post("/api/settings", json={
            "captioningDefaults": {"backend": "gemini_api", "modelSize": "8B"},
        })
        assert r.status_code == 200

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.captioning_defaults.backend == "gemini_api"
        assert loaded.state.app_settings.captioning_defaults.model_size == "8B"


class TestSettingsSchemaDrift:
    def test_update_request_tracks_app_settings_fields(self):
        assert set(AppSettings.model_fields) == set(UpdateSettingsRequest.model_fields)

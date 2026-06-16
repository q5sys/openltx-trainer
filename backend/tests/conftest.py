"""Test infrastructure for backend integration-style endpoint tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app_factory import create_app
from app_handler import ServiceBundle
from runtime_config.model_download_specs import (
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    resolve_model_path,
)
from runtime_config.port_constant import PORT
from state import RuntimeConfig, build_initial_state, set_state_service_for_tests
from state.app_settings import AppSettings
from state.app_state_types import HfAuthenticated
from tests.fakes.services import FakeServices

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture"
)

DEFAULT_APP_SETTINGS = AppSettings()


@pytest.fixture
def fake_services() -> FakeServices:
    return FakeServices()


@pytest.fixture(autouse=True)
def test_state(tmp_path: Path, fake_services: FakeServices):
    """Provide a fresh AppHandler per test and register it in DI."""
    app_data = tmp_path / "app_data"
    default_models_dir = app_data / "models"
    outputs_dir = tmp_path / "outputs"

    for directory in (default_models_dir, outputs_dir, app_data):
        directory.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig(
        device=torch.device("cpu"),
        app_data_dir=app_data,
        default_models_dir=default_models_dir,
        outputs_dir=outputs_dir,
        settings_file=app_data / "settings.json",
        local_generations_mode="full_models_loading",
        use_sage_attention=False,
        camera_motion_prompts={},
        default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        dev_mode=False,
        hf_oauth_client_id="test-client-id",
        backend_port=PORT,
    )

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

    handler = build_initial_state(
        config,
        DEFAULT_APP_SETTINGS.model_copy(deep=True),
        service_bundle=bundle,
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    set_state_service_for_tests(handler)
    yield handler


TEST_ADMIN_TOKEN = "test-admin-token"


@pytest.fixture
def client(test_state):
    from starlette.testclient import TestClient

    app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def default_app_settings() -> AppSettings:
    return DEFAULT_APP_SETTINGS.model_copy(deep=True)


def _test_model_path(test_state, cp_id: str) -> Path:
    return resolve_model_path(test_state.config.default_models_dir, cp_id)


@pytest.fixture
def create_fake_model_files(test_state):
    def _create():
        ltx_spec = get_ltx_model_spec(get_latest_ltx_model_id())

        for cp_id in (ltx_spec.model_cp, ltx_spec.upscale_cp):
            path = _test_model_path(test_state, cp_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)

        te_dir = _test_model_path(test_state, ltx_spec.text_encoder_cp)
        te_dir.mkdir(parents=True, exist_ok=True)
        (te_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        (te_dir / "tokenizer.model").write_bytes(b"\x00" * 1024)

    return _create

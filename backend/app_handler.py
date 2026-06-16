"""Application state composition root and dependency wiring."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from state.app_settings import AppSettings
from handlers import (
    CaptionHandler,
    DatasetHandler,
    DatasetValidationHandler,
    DownloadHandler,
    HealthHandler,
    HuggingFaceAuthHandler,
    ModelsHandler,
    RuntimePolicyHandler,
    SettingsHandler,
    TrainingHandler,
    VerificationHandler,
)
from runtime_config.runtime_config import RuntimeConfig
from services.caption_pipeline.caption_pipeline import CaptionPipeline
from services.dataset_pipeline.dataset_pipeline import DatasetPipeline
from services.training_supervisor.training_supervisor import TrainingSupervisor
from services.verification_pipeline.verification_pipeline import VerificationPipeline
from services.interfaces import (
    GpuCleaner,
    GpuInfo,
    HTTPClient,
    ModelDownloader,
    TaskRunner,
    TextEncoder,
    VideoProcessor,
)
from state.app_state_types import AppState, TextEncoderState


class AppHandler:
    """Composition-only state service exposing typed domain handlers."""

    def __init__(
        self,
        config: RuntimeConfig,
        default_settings: AppSettings,
        http: HTTPClient,
        gpu_cleaner: GpuCleaner,
        model_downloader: ModelDownloader,
        gpu_info: GpuInfo,
        video_processor: VideoProcessor,
        text_encoder: TextEncoder,
        task_runner: TaskRunner,
        dataset_pipeline: DatasetPipeline,
        caption_pipeline: CaptionPipeline,
        training_supervisor: TrainingSupervisor,
        verification_pipeline: VerificationPipeline,
    ) -> None:
        self.config = config

        # Exposed for tests and diagnostics.
        self.http = http
        self.gpu_cleaner = gpu_cleaner
        self.model_downloader = model_downloader
        self.gpu_info = gpu_info
        self.video_processor = video_processor
        self.task_runner = task_runner

        self._lock = threading.RLock()

        self.state = AppState(
            downloading_session=None,
            text_encoder=TextEncoderState(service=text_encoder),
            app_settings=default_settings.model_copy(deep=True),
        )

        # ============================================================
        # Handlers (wired in dependency order)
        # ============================================================

        self.settings = SettingsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.models = ModelsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.hf_auth = HuggingFaceAuthHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.downloads = DownloadHandler(
            state=self.state,
            lock=self._lock,
            models_handler=self.models,
            model_downloader=model_downloader,
            task_runner=task_runner,
            config=config,
        )

        self.health = HealthHandler(
            state=self.state,
            lock=self._lock,
            models_handler=self.models,
            gpu_info=gpu_info,
            config=config,
        )

        self.dataset = DatasetHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            dataset_pipeline=dataset_pipeline,
        )

        self.dataset_validation = DatasetValidationHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            dataset_pipeline=dataset_pipeline,
        )

        self.caption = CaptionHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            caption_pipeline=caption_pipeline,
        )

        self.training = TrainingHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            training_supervisor=training_supervisor,
            gpu_info=gpu_info,
        )


        self.verification = VerificationHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            verification_pipeline=verification_pipeline,
        )

        self.runtime_policy = RuntimePolicyHandler(config=config)

        self.downloads.cleanup_downloading_dir()

        self.load_persistent_state(default_settings)

    def load_persistent_state(self, default_settings: AppSettings) -> None:
        """Load persisted state from disk (settings, HF auth token, etc.)."""
        self.settings.load_settings(default_settings)
        self.hf_auth.load_token()


@dataclass
class ServiceBundle:
    http: HTTPClient
    gpu_cleaner: GpuCleaner
    model_downloader: ModelDownloader
    gpu_info: GpuInfo
    video_processor: VideoProcessor
    text_encoder: TextEncoder
    task_runner: TaskRunner
    dataset_pipeline: DatasetPipeline
    caption_pipeline: CaptionPipeline
    training_supervisor: TrainingSupervisor
    verification_pipeline: VerificationPipeline


def build_default_service_bundle(config: RuntimeConfig) -> ServiceBundle:
    """Build real runtime services with lazy heavy imports isolated from tests."""
    from services.gpu_cleaner.torch_cleaner import TorchCleaner
    from services.gpu_info.gpu_info_impl import GpuInfoImpl
    from services.http_client.http_client_impl import HTTPClientImpl
    from services.model_downloader.hugging_face_downloader import HuggingFaceDownloader
    from services.task_runner.threading_runner import ThreadingRunner
    from services.text_encoder.ltx_text_encoder import LTXTextEncoder
    from services.video_processor.video_processor_impl import VideoProcessorImpl

    http = HTTPClientImpl()

    from services.caption_pipeline.caption_pipeline_impl import CaptionPipelineImpl
    from services.dataset_pipeline.dataset_pipeline_impl import DatasetPipelineImpl

    return ServiceBundle(
        http=http,
        gpu_cleaner=TorchCleaner(device=config.device),
        model_downloader=HuggingFaceDownloader(),
        gpu_info=GpuInfoImpl(),
        video_processor=VideoProcessorImpl(),
        text_encoder=LTXTextEncoder(
            device=config.device,
            http=http,
        ),
        task_runner=ThreadingRunner(),
        dataset_pipeline=DatasetPipelineImpl(),
        caption_pipeline=CaptionPipelineImpl(),
        training_supervisor=_build_training_supervisor(config),
        verification_pipeline=_build_verification_pipeline(config),
    )


def build_initial_state(
    config: RuntimeConfig,
    default_settings: AppSettings,
    service_bundle: ServiceBundle | None = None,
) -> AppHandler:
    bundle = service_bundle or build_default_service_bundle(config)

    return AppHandler(
        config=config,
        default_settings=default_settings,
        http=bundle.http,
        gpu_cleaner=bundle.gpu_cleaner,
        model_downloader=bundle.model_downloader,
        gpu_info=bundle.gpu_info,
        video_processor=bundle.video_processor,
        text_encoder=bundle.text_encoder,
        task_runner=bundle.task_runner,
        dataset_pipeline=bundle.dataset_pipeline,
        caption_pipeline=bundle.caption_pipeline,
        training_supervisor=bundle.training_supervisor,
        verification_pipeline=bundle.verification_pipeline,
    )


def _build_training_supervisor(config: RuntimeConfig) -> TrainingSupervisor:
    """Build the real training supervisor, using the app data directory."""
    from services.training_supervisor.training_supervisor_impl import TrainingSupervisorImpl

    jobs_root = config.default_models_dir.parent
    return TrainingSupervisorImpl(jobs_root=jobs_root)


def _build_verification_pipeline(config: RuntimeConfig) -> VerificationPipeline:
    """Build the real verification pipeline stub.

    The real GPU implementation will be added when the LTX model
    loading and LORA stack logic are implemented. For now this
    returns the fake pipeline so the app can boot and the UI can
    be developed against it.
    """
    from services.verification_pipeline.fake_verification_pipeline import FakeVerificationPipeline

    jobs_root = config.default_models_dir.parent
    return FakeVerificationPipeline(jobs_root=jobs_root)

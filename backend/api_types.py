"""Pydantic request/response models and typed aliases for the backend server."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

ModelCheckpointID = Literal[
    "ltx-2.3-22b-dev",
    "ltx-2.3-spatial-upscaler-x2-1.0",
    "gemma-3-12b-it-qat-q4_0-unquantized",
    "qwen3-vl-4b-instruct",
]
LTXLocalModelId = Literal["ltx-2.3-22b-dev"]


JsonObject: TypeAlias = dict[str, object]


# ============================================================
# Response Models
# ============================================================


class ModelStatusItem(BaseModel):
    id: str
    name: str
    loaded: bool
    downloaded: bool


class GpuTelemetry(BaseModel):
    name: str
    vram: int
    vramUsed: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    models_loaded: bool
    active_model: str | None
    gpu_info: GpuTelemetry
    sage_attention: bool
    models_status: list[ModelStatusItem]


class GpuDeviceItem(BaseModel):
    index: int
    name: str


class GpuListResponse(BaseModel):
    devices: list[GpuDeviceItem]


class GpuMemoryResponse(BaseModel):
    # Live per-device VRAM for one GPU index, in MB. ``available`` is
    # False when the device cannot be queried; the UI then hides the
    # readout instead of showing zeros.
    available: bool
    total_mb: int
    used_mb: int



class GpuInfoResponse(BaseModel):
    cuda_available: bool
    mps_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None
    vram_gb: int | None
    gpu_info: GpuTelemetry



# ============================================================
# Stage F: low-VRAM auto-tune
# ============================================================


# Mirror of ``training_worker.config.LowVramMode``. Duplicated here so
# the FastAPI app does not have to import the training_worker package
# at module load time (the worker package imports torch).
LowVramModeApi = Literal["off", "fp8", "nf4"]

# Mirror of ``training_worker.engine.gpu_budget.TrainingProfile`` and
# ``training_worker.config.TrainingConfig.profile``. Duplicated here to
# keep the FastAPI app free of the torch-importing worker package.
TrainingProfileApi = Literal["image", "video"]

RecommendationConfidenceApi = Literal[
    "baseline", "supported", "plausible", "unsupported"
]


class AutoTuneVramRequest(BaseModel):
    """Request body for ``POST /api/training/auto-tune-vram``.

    All fields are optional. When the hardware fields are omitted the
    backend queries ``GpuInfo`` for VRAM and ``psutil.virtual_memory()``
    for host RAM. Supplying them explicitly is the entry point the
    Stage F smoke script uses to simulate a smaller card on the 5090.
    """

    vram_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Override the detected GPU VRAM in bytes. Used by the "
            "Stage F smoke script to simulate a 24/20/16 GB card on "
            "a real 32 GB 5090."
        ),
    )
    system_ram_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Override the detected host RAM in bytes. Mostly useful "
            "in tests."
        ),
    )
    profile: TrainingProfileApi = Field(
        default="video",
        description=(
            "Training profile to tune for. The image profile (frames=1) "
            "and video profile (121 frames) have different VRAM curves "
            "and therefore different tier tables. Defaults to 'video' "
            "to match TrainingConfig.profile."
        ),
    )



class AutoTuneVramResponse(BaseModel):
    """One row of the feasibility table returned to the Training UI.

    Mirrors ``training_worker.engine.gpu_budget.LowVramRecommendation``
    one-to-one. The frontend renders ``tier_label`` and ``warning``
    verbatim and binds the three knob fields to the start-job form.
    """

    tier_label: str
    low_vram_mode: LowVramModeApi
    blocks_resident_on_gpu: int
    gradient_checkpointing: bool
    estimated_peak_vram_gb: float
    estimated_throughput_multiplier: float
    required_host_ram_gb: int
    confidence: RecommendationConfidenceApi
    warning: str = ""
    detected_vram_bytes: int
    detected_system_ram_bytes: int


# Mirror of ``training_worker.engine.vram_sweep_data.SweepQuant``.
SweepQuantApi = Literal["nf4", "fp8", "bf16"]


class VramSweepCellResponse(BaseModel):
    """One measured cell of the VRAM benchmark sweep.

    Mirrors ``training_worker.engine.vram_sweep_data.VramSweepCell``.
    The Training UI renders the full list as a sortable table so the
    operator can pick any (quant, blocks_resident) combination, not
    just the auto-tune recommendation.
    """

    profile: TrainingProfileApi
    quant: SweepQuantApi
    blocks_resident_on_gpu: int
    peak_vram_gb: float
    runtime_s: int


class VramSweepResponse(BaseModel):
    """The full measured VRAM sweep plus provenance for the UI."""

    source: str
    total_blocks: int
    cells: list[VramSweepCellResponse]



class RuntimePolicyResponse(BaseModel):

    force_api_generations: bool


class DownloadProgressRunningResponse(BaseModel):
    status: Literal["downloading"]
    current_downloading_file: ModelCheckpointID | None
    current_file_progress: float
    total_progress: float
    total_downloaded_bytes: int
    expected_total_bytes: int
    completed_files: set[ModelCheckpointID]
    all_files: set[ModelCheckpointID]
    error: None = None
    speed_bytes_per_sec: float


class DownloadProgressCompleteResponse(BaseModel):
    status: Literal["complete"]


class DownloadProgressErrorResponse(BaseModel):
    status: Literal["error"]
    error: str


DownloadProgressResponse: TypeAlias = (
    DownloadProgressRunningResponse | DownloadProgressCompleteResponse | DownloadProgressErrorResponse
)


# ============================================================
# HuggingFace auth
# ============================================================


class HuggingFaceLoginResponse(BaseModel):
    client_id: str
    redirect_uri: str
    scope: str
    state: str
    code_challenge: str
    code_challenge_method: str


class HuggingFaceAuthStatusResponse(BaseModel):
    status: Literal["authenticated", "pending", "not_authenticated"]


class HuggingFaceLogoutResponse(BaseModel):
    status: Literal["logged_out"]


class ModelDownloadStartResponse(BaseModel):
    status: Literal["started"]
    message: str
    sessionId: str


class LtxDownloadRecommendationResponse(BaseModel):
    status: Literal["download"]
    cps_to_download: list[ModelCheckpointID]


class LtxUpgradeRecommendationResponse(BaseModel):
    status: Literal["upgrade"]
    ltx_model_id: LTXLocalModelId
    upgrade_message: str | None = None
    cps_to_download: list[ModelCheckpointID]
    cps_to_delete: list[ModelCheckpointID]


class LtxOkRecommendationResponse(BaseModel):
    status: Literal["ok"]


LtxRecommendationResponse: TypeAlias = (
    LtxDownloadRecommendationResponse | LtxUpgradeRecommendationResponse | LtxOkRecommendationResponse
)


class TextEncoderRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None
    expected_size_bytes: int
    expected_size_gb: float


class StatusResponse(BaseModel):
    status: str


class HTTPErrorResponse(BaseModel):
    code: str
    message: str


# ============================================================
# Request Models
# ============================================================


def _default_model_types() -> set[ModelCheckpointID]:
    return set()


class ModelDownloadRequest(BaseModel):
    type: Literal["download", "upgrade"] = "download"
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


ModelAccessStatus: TypeAlias = Literal["authorized", "not_authorized"]


class CheckModelAccessRequest(BaseModel):
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


class CheckModelAccessResponse(BaseModel):
    access: dict[str, ModelAccessStatus]


class ModelDeleteRequest(BaseModel):
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)

"""Canonical app settings schema and patch models."""

from __future__ import annotations

from typing import Any, TypeGuard, TypeVar, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator


def _to_camel_case(field_name: str) -> str:
    head, *tail = field_name.split("_")
    return head + "".join(part.title() for part in tail)


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default

    parsed = int(value)
    return max(minimum, min(maximum, parsed))


def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    if value is None:
        return default

    parsed = float(value)
    return max(minimum, min(maximum, parsed))


class SettingsBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore",
    )


class SettingsPatchModel(SettingsBaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )


class ModelDirs(SettingsBaseModel):
    base_models: str = "auto"
    captioner: str = "auto"
    trained_loras: str = "auto"


class TrainingDefaults(SettingsBaseModel):
    save_optimizer_state: bool = True
    keep_last_n_checkpoints: int = 0
    sample_on_save: bool = True
    auto_advance_phases: bool = True
    transformer_quantization: str = "float8"
    text_encoder_quantization: str = "float8"

    @field_validator("keep_last_n_checkpoints", mode="before")
    @classmethod
    def _clamp_keep_last_n(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=100, default=0)


class CaptioningDefaults(SettingsBaseModel):
    backend: str = "qwen_vl_local"
    model_family: str = "qwen3-vl"
    model_size: str = "4B"
    abliterated: bool = False
    quantization: str = "fp16"
    captioner_idle_timeout_seconds: int = 300

    @field_validator("captioner_idle_timeout_seconds", mode="before")
    @classmethod
    def _clamp_timeout(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=3600, default=300)


class OpenAICompatible(SettingsBaseModel):
    base_url: str = ""
    api_key: str = ""


class CaptioningApiKeys(SettingsBaseModel):
    gemini: str = ""
    openai: str = ""
    anthropic: str = ""
    openai_compatible: OpenAICompatible = Field(default_factory=OpenAICompatible)


class VerificationDefaults(SettingsBaseModel):
    default_cfg: float = 10.0
    default_frames: int = 49
    default_size: list[int] = Field(default_factory=lambda: [512, 512])

    @field_validator("default_cfg", mode="before")
    @classmethod
    def _clamp_cfg(cls, value: Any) -> float:
        return _clamp_float(value, minimum=1.0, maximum=30.0, default=10.0)

    @field_validator("default_frames", mode="before")
    @classmethod
    def _clamp_frames(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=257, default=49)


class AppSettings(SettingsBaseModel):
    keep_models_loaded: bool = True
    use_torch_compile: bool = False
    load_on_startup: bool = False
    default_gpu_index: int = 0
    model_dirs: ModelDirs = Field(default_factory=ModelDirs)
    training_defaults: TrainingDefaults = Field(default_factory=TrainingDefaults)
    captioning_defaults: CaptioningDefaults = Field(default_factory=CaptioningDefaults)
    captioning_api_keys: CaptioningApiKeys = Field(default_factory=CaptioningApiKeys)
    verification_defaults: VerificationDefaults = Field(default_factory=VerificationDefaults)

    @field_validator("default_gpu_index", mode="before")
    @classmethod
    def _clamp_gpu_index(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=15, default=0)


SettingsModelT = TypeVar("SettingsModelT", bound=SettingsBaseModel)
_PARTIAL_MODEL_CACHE: dict[type[SettingsBaseModel], type[SettingsPatchModel]] = {}


def _wrap_optional(annotation: Any) -> Any:
    if type(None) in get_args(annotation):
        return annotation
    return annotation | None


def _to_partial_annotation(annotation: Any) -> Any:
    if _is_settings_model_annotation(annotation):
        return make_partial_model(annotation)
    return annotation


def make_partial_model(model: type[SettingsModelT]) -> type[SettingsPatchModel]:
    cached = _PARTIAL_MODEL_CACHE.get(model)
    if cached is not None:
        return cached

    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_info in model.model_fields.items():
        partial_annotation = _wrap_optional(_to_partial_annotation(field_info.annotation))
        fields[field_name] = (partial_annotation, Field(default=None))

    partial_model = create_model(
        f"{model.__name__}Patch",
        __base__=SettingsPatchModel,
        **cast(Any, fields),
    )

    _PARTIAL_MODEL_CACHE[model] = partial_model
    return partial_model


def _is_settings_model_annotation(annotation: object) -> TypeGuard[type[SettingsBaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, SettingsBaseModel)


AppSettingsPatch = make_partial_model(AppSettings)
UpdateSettingsRequest = AppSettingsPatch


def _mask_api_key(key: str) -> str:
    """Mask an API key for display, showing only the last 4 characters."""
    if len(key) <= 4:
        return "*" * len(key)
    return "*" * (len(key) - 4) + key[-4:]


class MaskedOpenAICompatible(SettingsBaseModel):
    base_url: str = ""
    api_key: str = ""


class MaskedCaptioningApiKeys(SettingsBaseModel):
    gemini: str = ""
    openai: str = ""
    anthropic: str = ""
    openai_compatible: MaskedOpenAICompatible = Field(default_factory=MaskedOpenAICompatible)


class SettingsResponse(SettingsBaseModel):
    keep_models_loaded: bool = True
    use_torch_compile: bool = False
    load_on_startup: bool = False
    default_gpu_index: int = 0
    model_dirs: ModelDirs = Field(default_factory=ModelDirs)
    training_defaults: TrainingDefaults = Field(default_factory=TrainingDefaults)
    captioning_defaults: CaptioningDefaults = Field(default_factory=CaptioningDefaults)
    captioning_api_keys: MaskedCaptioningApiKeys = Field(default_factory=MaskedCaptioningApiKeys)
    verification_defaults: VerificationDefaults = Field(default_factory=VerificationDefaults)


def to_settings_response(settings: AppSettings) -> SettingsResponse:
    data: dict[str, Any] = settings.model_dump(by_alias=False)
    # Mask API keys before returning
    keys = cast(dict[str, Any], data.get("captioning_api_keys", {}))
    for field_name in ("gemini", "openai", "anthropic"):
        raw = str(keys.get(field_name, ""))
        if raw:
            keys[field_name] = _mask_api_key(raw)
    compat = cast(dict[str, Any], keys.get("openai_compatible", {}))
    api_key = str(compat.get("api_key", ""))
    if api_key:
        compat["api_key"] = _mask_api_key(api_key)
    return SettingsResponse.model_validate(data)

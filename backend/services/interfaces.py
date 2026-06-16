"""Compatibility re-exports for service interfaces."""

from __future__ import annotations

from services.gpu_cleaner.gpu_cleaner import GpuCleaner
from services.gpu_info.gpu_info import GpuInfo, GpuMemoryPayload, GpuTelemetryPayload

from services.http_client.http_client import HTTPClient, HttpResponseLike, HttpTimeoutError
from services.model_downloader.model_downloader import ModelDownloader
from services.services_utils import JSONScalar, JSONValue
from services.task_runner.task_runner import TaskRunner
from services.text_encoder.text_encoder import TextEncoder
from services.dataset_pipeline.dataset_pipeline import DatasetPipeline
from services.video_processor.video_processor import VideoInfoPayload, VideoProcessor

__all__ = [
    "JSONScalar",
    "JSONValue",
    "GpuTelemetryPayload",
    "GpuMemoryPayload",

    "VideoInfoPayload",
    "HttpTimeoutError",
    "HttpResponseLike",
    "HTTPClient",
    "ModelDownloader",
    "GpuCleaner",
    "GpuInfo",
    "VideoProcessor",
    "TaskRunner",
    "TextEncoder",
    "DatasetPipeline",
]

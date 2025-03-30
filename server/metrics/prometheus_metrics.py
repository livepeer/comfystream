"""Manages Prometheus metrics collection and exposure."""

from prometheus_client import Gauge, generate_latest
from aiohttp import web
from typing import Optional


class MetricsManager:
    """Manages Prometheus metrics collection."""

    def __init__(self, include_stream_id: bool = False):
        """Initializes the MetricsManager class.

        Args:
            include_stream_id: Whether to include the stream ID as a label in the
                metrics.
        """
        self._enabled = False
        self._include_stream_id = include_stream_id

        base_labels = ["stream_id"] if include_stream_id else []
        self._fps_gauge = Gauge(
            "stream_fps", "Frames per second of the stream", base_labels
        )
        self._startup_time_gauge = Gauge(
            "stream_startup_time",
            "Time taken to start the stream",
            base_labels,
        )
        self._pipeline_audio_warmup_time_gauge = Gauge(
            "stream_pipeline_audio_warmup_time",
            "Time taken to warm up the audio pipeline",
            base_labels,
        )
        self._pipeline_video_warmup_time_gauge = Gauge(
            "stream_pipeline_video_warmup_time",
            "Time taken to warm up the video pipeline",
            base_labels,
        )

    def enable(self):
        """Enable Prometheus metrics collection."""
        self._enabled = True

    def update_fps(self, fps: float, stream_id: Optional[str] = None):
        """Update fps metrics for a given stream.

        Args:
            fps: The current frames per second.
            stream_id: The ID of the stream.
        """
        if self._enabled:
            if self._include_stream_id:
                self._fps_gauge.labels(stream_id=stream_id or "").set(fps)
            else:
                self._fps_gauge.set(fps)

    def update_startup_time(self, startup_time: float, stream_id: Optional[str] = None):
        """Update startup time metrics for a given stream.

        Args:
            startup_time: The time taken to start the stream.
            stream_id: The ID of the stream.
        """
        if self._enabled:
            if self._include_stream_id:
                self._startup_time_gauge.labels(stream_id=stream_id or "").set(
                    startup_time
                )
            else:
                self._startup_time_gauge.set(startup_time)

    def update_video_warmup_time(
        self, warmup_time: float, stream_id: Optional[str] = None
    ):
        """Update video pipeline warmup time metrics for a given stream.

        Args:
            warmup_time: The time taken to warm up the video pipeline.
            stream_id: The ID of the stream.
        """
        if self._enabled:
            if self._include_stream_id:
                self._pipeline_video_warmup_time_gauge.labels(
                    stream_id=stream_id or ""
                ).set(warmup_time)
            else:
                self._pipeline_video_warmup_time_gauge.set(warmup_time)

    def update_audio_warmup_time(
        self, warmup_time: float, stream_id: Optional[str] = None
    ):
        """Update audio pipeline warmup time metrics for a given stream.

        Args:
            warmup_time: The time taken to warm up the audio pipeline.
            stream_id: The ID of the stream.
        """
        if self._enabled:
            if self._include_stream_id:
                self._pipeline_audio_warmup_time_gauge.labels(
                    stream_id=stream_id or ""
                ).set(warmup_time)
            else:
                self._pipeline_audio_warmup_time_gauge.set(warmup_time)

    async def metrics_handler(self, _):
        """Handle Prometheus metrics endpoint."""
        return web.Response(body=generate_latest(), content_type="text/plain")

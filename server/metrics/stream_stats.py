"""Handles real-time video stream statistics for JSON API publishing."""

from typing import Any, Dict, Optional
import json
from aiohttp import web
from aiortc import MediaStreamTrack
from utils.fps_meter import FPSMeter
from .prometheus_metrics import MetricsManager
from .pipeline_stats import PipelineStats
import time


class StreamStats:
    """Tracks real-time statistics of the stream.

    Attributes:
        fps_meter: The FPSMeter instance for the stream.
        start_timestamp: The timestamp when the stream started.
        pipeline_stats: The PipelineStats instance for the stream.
    """

    def __init__(self, track_id: str, metrics_manager: Optional[MetricsManager] = None):
        """Initializes the StreamStats class.

        Args:
            track_id: The ID of the stream track.
            metrics_manager: The Prometheus metrics manager instance.
        """
        update_metrics_callback = (
            metrics_manager.update_fps if metrics_manager else None
        )
        self.fps_meter = FPSMeter(
            track_id=track_id,
            update_metrics_callback=update_metrics_callback,
        )
        self.pipeline = PipelineStats(
            metrics_manager=metrics_manager, track_id=track_id
        )

        self.start_timestamp = None

        self._metrics_manager = metrics_manager
        self._startup_time = None

    @property
    def startup_time(self) -> float:
        """Time taken to start the stream."""
        return self._startup_time

    @startup_time.setter
    def startup_time(self, value: float):
        """Sets the time taken to start the stream."""
        if self._metrics_manager:
            self._metrics_manager.update_startup_time(value, self.fps_meter.track_id)
        self._startup_time = value

    async def get_fps(self) -> float:
        """Current frames per second (FPS) of the stream.

        Alias for FPSMeter's get_fps method.
        """
        return await self.fps_meter.get_fps()

    async def get_fps_measurements(self) -> list:
        """List of FPS measurements over time.

        Alias for FPSMeter's get_fps_measurements method.
        """
        return await self.fps_meter.get_fps_measurements()

    async def get_average_fps(self) -> float:
        """Average FPS over the last minute.

        Alias for FPSMeter's get_average_fps method.
        """
        return await self.fps_meter.get_average_fps()

    async def get_last_fps_calculation_time(self) -> float:
        """Timestamp of the last FPS calculation.

        Alias for FPSMeter's get_last_fps_calculation_time method.
        """
        return await self.fps_meter.get_last_fps_calculation_time()

    @property
    def time(self) -> float:
        """Elapsed time since the stream started."""
        return (
            0.0
            if self.start_timestamp is None
            else time.monotonic() - self.start_timestamp
        )

    async def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for easy JSON serialization."""
        return {
            "timestamp": self.time,
            "startup_time": self.startup_time,
            "pipeline": self.pipeline.to_dict(),
            "fps": await self.get_fps(),
            "minute_avg_fps": await self.get_average_fps(),
            "minute_fps_array": await self.get_fps_measurements(),
        }


class StreamStatsManager:
    """Handles real-time stream statistics collection."""

    def __init__(self, app: web.Application):
        """Initializes the StreamStatsManager class.

        Args:
            app: The web application instance storing stream tracks.
        """
        self._app = app

    async def collect_video_stats(
        self, video_track: MediaStreamTrack
    ) -> Dict[str, Any]:
        """Collects real-time statistics for a video track.

        Args:
            video_track: The video stream track instance.

        Returns:
            A dictionary containing FPS-related statistics.
        """
        return {"type": video_track.kind, **await video_track.stats.to_dict()}

    async def collect_audio_stats(
        self, audio_track: MediaStreamTrack
    ) -> Dict[str, Any]:
        """Collects real-time statistics for an audio track.

        Args:
            audio_track: The audio stream track instance.

        Returns:
            A dictionary containing audio-related statistics.
        """
        return {"type": audio_track.kind, **await audio_track.stats.to_dict()}

    async def collect_all_stream_stats(self, _) -> web.Response:
        """Retrieves real-time statistics for all active video and audio streams.

        Returns:
            A JSON response containing FPS statistics for all streams.
        """
        tracks = {
            **self._app.get("video_tracks", {}),
            **self._app.get("audio_tracks", {}),
        }
        all_stats = {
            stream_id: await (
                self.collect_video_stats(track)
                if track.kind == "video"
                else self.collect_audio_stats(track)
            )
            for stream_id, track in tracks.items()
        }

        return web.Response(
            content_type="application/json",
            text=json.dumps(all_stats),
        )

    async def collect_stream_stats_by_id(self, request: web.Request) -> web.Response:
        """Retrieves real-time statistics for a specific video or audio stream by ID.

        Args:
            request: The HTTP request containing the stream ID.

        Returns:
            A JSON response with stream statistics or an error message.
        """
        stream_id = request.match_info.get("stream_id")
        tracks = {
            **self._app.get("video_tracks", {}),
            **self._app.get("audio_tracks", {}),
        }
        track = tracks.get(stream_id)

        if not track:
            error_response = {"error": "Stream not found"}
            return web.Response(
                status=404,
                content_type="application/json",
                text=json.dumps(error_response),
            )

        stats = await (
            self.collect_video_stats(track)
            if track.kind == "video"
            else self.collect_audio_stats(track)
        )
        return web.Response(
            content_type="application/json",
            text=json.dumps(stats),
        )

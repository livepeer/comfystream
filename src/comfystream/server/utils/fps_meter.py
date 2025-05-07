"""Module to calculate and store the framerate of a stream by counting frames."""

import asyncio
import logging
import time
from collections import deque
from comfystream.server.metrics import MetricsManager

logger = logging.getLogger(__name__)


class FPSMeter:
    """Class to calculate and store the framerate of a stream by counting frames."""

    def __init__(self, metrics_manager: MetricsManager, track_id: str):
        """Initializes the FPSMeter class."""
        self._lock = asyncio.Lock()
        self._fps_interval_frame_count = 0
        self._last_fps_calculation_time = None
        self._fps_loop_start_time = None
        self._fps = 0.0
        self._fps_measurements = deque(maxlen=60)
        self._running_event = asyncio.Event()
        self._metrics_manager = metrics_manager
        self.track_id = track_id
        self._fps_task = None
        self._stop_event = asyncio.Event()

        self._fps_task = asyncio.create_task(self._calculate_fps_loop())

    async def _calculate_fps_loop(self):
        """Loop to calculate FPS periodically."""
        try:
            # Wait for first frame with timeout
            try:
                await asyncio.wait_for(self._running_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("FPS meter timed out waiting for first frame")
                return

            self._fps_loop_start_time = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    async with self._lock:
                        current_time = time.monotonic()
                        if self._last_fps_calculation_time is not None:
                            time_diff = current_time - self._last_fps_calculation_time
                            self._fps = (
                                self._fps_interval_frame_count / time_diff
                                if time_diff > 0
                                else 0.0
                            )
                            self._fps_measurements.append(
                                {
                                    "timestamp": current_time - self._fps_loop_start_time,
                                    "fps": self._fps,
                                }
                            )

                        # Reset tracking variables for the next interval.
                        self._last_fps_calculation_time = current_time
                        self._fps_interval_frame_count = 0

                    # Update Prometheus metrics if enabled.
                    self._metrics_manager.update_fps_metrics(self._fps, self.track_id)

                    await asyncio.sleep(1)  # Calculate FPS every second.
                except Exception as e:
                    logger.error(f"Error in FPS calculation loop: {e}")
                    await asyncio.sleep(1)  # Wait before retrying
        except asyncio.CancelledError:
            logger.info("FPS calculation loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in FPS calculation loop: {e}")
        finally:
            self._stop_event.set()

    async def cleanup(self):
        """Clean up resources and stop the FPS calculation loop."""
        if self._fps_task:
            self._stop_event.set()
            try:
                await asyncio.wait_for(self._fps_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("FPS task cleanup timed out or was cancelled")
            self._fps_task = None

    async def increment_frame_count(self):
        """Increment the frame count to calculate FPS."""
        async with self._lock:
            self._fps_interval_frame_count += 1
            if not self._running_event.is_set():
                self._running_event.set()

    @property
    async def fps(self) -> float:
        """Get the current output frames per second (FPS).

        Returns:
            The current output FPS.
        """
        async with self._lock:
            return self._fps

    @property
    async def fps_measurements(self) -> list:
        """Get the array of FPS measurements for the last minute.

        Returns:
            The array of FPS measurements for the last minute.
        """
        async with self._lock:
            return list(self._fps_measurements)

    @property
    async def average_fps(self) -> float:
        """Calculate the average FPS from the measurements taken in the last minute.

        Returns:
            The average FPS over the last minute.
        """
        async with self._lock:
            return (
                sum(m["fps"] for m in self._fps_measurements)
                / len(self._fps_measurements)
                if self._fps_measurements
                else self._fps
            )

    @property
    async def last_fps_calculation_time(self) -> float:
        """Get the elapsed time since the last FPS calculation.

        Returns:
            The elapsed time in seconds since the last FPS calculation.
        """
        async with self._lock:
            if (
                self._last_fps_calculation_time is None
                or self._fps_loop_start_time is None
            ):
                return 0.0
            return self._last_fps_calculation_time - self._fps_loop_start_time

import asyncio
import json
import logging
import os
from typing import List, Optional

from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from pytrickle.video_utils import create_loading_frame
from comfystream.pipeline import Pipeline
from comfystream.utils import convert_prompt, ComfyStreamParamsUpdateRequest, get_default_workflow
import av

logger = logging.getLogger(__name__)


class ComfyStreamFrameProcessor(FrameProcessor):
    """
    Integrated ComfyStream FrameProcessor for pytrickle.

    This class wraps the ComfyStream Pipeline to work with pytrickle's streaming architecture.
    """

    def __init__(self, text_poll_interval: float = 0.25, **load_params):
        """Initialize with load parameters for pipeline creation.

        Args:
            text_poll_interval: Interval in seconds to poll for text outputs (default: 0.25)
            **load_params: Parameters for pipeline creation
        """
        self.pipeline = None
        self._load_params = load_params
        self._text_poll_interval = text_poll_interval
        self._stream_processor = None
        self._warmup_task = None
        self._text_forward_task = None
        self._background_tasks = []
        self._stop_event = asyncio.Event()
        self._runner_active = False
        # Loading overlay / warmup gating
        self._loading_active: bool = False
        self._warmup_done: asyncio.Event = asyncio.Event()
        self._loading_message: str = "Loading..."
        self._loading_progress: Optional[float] = None
        self._frame_counter: int = 0
        self._warmup_passthrough_enabled: bool = False
        super().__init__()

    def set_stream_processor(self, stream_processor):
        """Set reference to StreamProcessor for data publishing."""
        self._stream_processor = stream_processor
        logger.info("StreamProcessor reference set for text data publishing")

    def _setup_text_monitoring(self):
        """Set up background text forwarding from the pipeline."""
        try:
            if self.pipeline and self._stream_processor:
                # Reset stop event for new stream
                self._reset_stop_event()
                # Start forwarder only if workflow has text outputs (best-effort)
                should_start = True
                try:
                    should_start = bool(self.pipeline.produces_text_output())
                except Exception:
                    # If capability check fails, default to starting forwarder
                    should_start = True

                if should_start:
                    # Start a background task that forwards text outputs via StreamProcessor
                    if self._text_forward_task and not self._text_forward_task.done():
                        logger.debug("Text forwarder already running; not starting another")
                        return

                    async def _forward_text_loop():
                        try:
                            logger.info("Starting background text forwarder task")
                            while not self._stop_event.is_set():
                                try:
                                    # Non-blocking poll; sleep if no text to avoid tight loop
                                    text = await self.pipeline.get_text_output()
                                    if text is None or text.strip() == "":
                                        await asyncio.sleep(self._text_poll_interval)
                                        continue
                                    if self._stream_processor:
                                        success = await self._stream_processor.send_data(text)
                                        if not success:
                                            logger.debug("Text send failed; stopping text forwarder")
                                            break
                                except asyncio.CancelledError:
                                    logger.debug("Text forwarder task cancelled")
                                    raise
                        except asyncio.CancelledError:
                            # Propagate to finally for cleanup
                            raise
                        except Exception as e:
                            logger.error(f"Error in text forwarder: {e}")
                        finally:
                            logger.info("Text forwarder task exiting")

                    self._text_forward_task = asyncio.create_task(_forward_text_loop())
                    self._background_tasks.append(self._text_forward_task)
        except Exception:
            logger.warning("Failed to set up text monitoring", exc_info=True)

    async def _stop_text_forwarder(self) -> None:
        """Stop the background text forwarder task if running."""
        task = self._text_forward_task
        if task and not task.done():
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Error while awaiting text forwarder cancellation", exc_info=True)
        self._text_forward_task = None

    async def on_stream_stop(self):
        """Called when stream stops - cleanup background tasks."""
        logger.info("Stream stopped, cleaning up background tasks")

        # Set stop event to signal all background tasks to stop
        self._stop_event.set()

        # Stop the ComfyStream client's prompt execution immediately to avoid no-input logs
        if self.pipeline and self.pipeline.client:
            logger.info("Stopping ComfyStream client prompt execution")
            try:
                await self.pipeline.client.stop_prompts_immediately()
            except Exception as e:
                logger.error(f"Error stopping ComfyStream client: {e}")
        self._runner_active = False

        # Stop text forwarder
        await self._stop_text_forwarder()

        # Cancel warmup if running and reset overlay state
        try:
            if self._warmup_task and not self._warmup_task.done():
                self._warmup_task.cancel()
                await self._warmup_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Warmup task cancellation error", exc_info=True)
        finally:
            self._loading_active = False
            self._warmup_done.set()

        # Cancel any other background tasks started by this processor
        for task in list(self._background_tasks):
            try:
                if task and not task.done():
                    task.cancel()
            except Exception:
                continue

        # Await task cancellations
        for task in list(self._background_tasks):
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug("Background task raised during shutdown", exc_info=True)

        self._background_tasks.clear()
        logger.info("All background tasks cleaned up")

    def _reset_stop_event(self):
        """Reset the stop event for a new stream."""
        self._stop_event.clear()


    async def _start_warmup_sequence(self) -> None:
        """Start (or restart) the warmup routine with loading overlay management."""
        try:
            if self._warmup_task and not self._warmup_task.done():
                self._warmup_task.cancel()
                try:
                    await self._warmup_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug("Previous warmup task cleanup error", exc_info=True)
        except Exception:
            logger.debug("Error cancelling prior warmup task", exc_info=True)

        self._frame_counter = 0
        self._loading_active = True
        self._warmup_done.clear()

        async def _warmup_and_finish():
            try:
                await self.warmup()
            except Exception:
                logger.debug("Warmup failed while running warmup sequence", exc_info=True)
            finally:
                self._loading_active = False
                self._warmup_done.set()

        self._warmup_task = asyncio.create_task(_warmup_and_finish())

    async def _ensure_runner_active(self) -> None:
        """Ensure the prompt runner is active when real frames arrive."""
        if not self.pipeline or not getattr(self.pipeline, "client", None):
            return
        if not self._runner_active:
            await self.pipeline.client.ensure_prompt_tasks_running()
            self.pipeline.client.resume()
            self._runner_active = True

    def _build_loading_overlay_frame(self, frame: VideoFrame) -> VideoFrame:
        """Render a loading overlay frame while warmup is in progress."""
        try:
            self._frame_counter += 1
            # Determine output dimensions—prefer pipeline target size when available
            width = getattr(self.pipeline, "width", None) or frame.tensor.shape[-2]
            height = getattr(self.pipeline, "height", None) or frame.tensor.shape[-3]

            overlay_np = create_loading_frame(
                width,
                height,
                message=self._loading_message,
                frame_counter=self._frame_counter,
                progress=self._loading_progress,
                color_format="RGB",
            )

            overlay_av = av.VideoFrame.from_ndarray(overlay_np, format="rgb24")
            overlay_av.pts = frame.timestamp
            overlay_av.time_base = frame.time_base

            overlay_frame = VideoFrame.from_av_frame_with_timing(overlay_av, frame)
            overlay_frame.side_data = frame.side_data
            return overlay_frame
        except Exception:
            logger.debug("Failed to generate loading overlay frame", exc_info=True)
            return frame

    async def load_model(self, **kwargs):
        """Load model, initialize pipeline, set default workflow once, and warm up."""
        params = {**self._load_params, **kwargs}

        if self.pipeline is None:
            self.pipeline = Pipeline(
                width=int(params.get('width', 512)),
                height=int(params.get('height', 512)),
                cwd=params.get('workspace', os.getcwd()),
                disable_cuda_malloc=params.get('disable_cuda_malloc', True),
                gpu_only=params.get('gpu_only', True),
                preview_method=params.get('preview_method', 'none'),
                comfyui_inference_log_level=params.get('comfyui_inference_log_level', "INFO"),
                logging_level=params.get('comfyui_inference_log_level', "INFO"),
                blacklist_custom_nodes=["ComfyUI-Manager"],
            )

        # Only set the default workflow if no prompts are currently configured
        has_prompts = False
        try:
            has_prompts = bool(getattr(self.pipeline.client, "current_prompts", []))
        except Exception:
            has_prompts = False

        if not has_prompts:
            default_workflow = get_default_workflow()
            # Apply default prompt without scheduling background warmup,
            # then perform warmup synchronously so server state remains LOADING
            await self._process_prompts(default_workflow, skip_warmup=True)
            if not self._loading_active:
                await self.warmup()
        else:
            # Prompts exist; perform warmup synchronously
            if not self._loading_active:
                await self.warmup()

    async def warmup(self):
        """Warm up the pipeline."""
        if not self.pipeline:
            logger.warning("Warmup requested before pipeline initialization")
            return

        logger.info("Running pipeline warmup...")
        try:
            # Ensure runner exists and is enabled for warmup
            await self.pipeline.client.ensure_prompt_tasks_running()
            self.pipeline.client.resume()

            capabilities = self.pipeline.get_workflow_io_capabilities()
            logger.info(f"Detected I/O capabilities: {capabilities}")

            if capabilities.get("video", {}).get("input") or capabilities.get("video", {}).get("output"):
                await self.pipeline.warm_video()

            if capabilities.get("audio", {}).get("input") or capabilities.get("audio", {}).get("output"):
                await self.pipeline.warm_audio()

        except Exception as e:
            logger.error(f"Warmup failed: {e}")
        finally:
            # Pause prompt loop after warmup; will resume on first real input
            try:
                self.pipeline.client.pause()
            except Exception:
                logger.debug("Failed to stop prompt loop after warmup", exc_info=True)
            self._runner_active = False
            if not self._loading_active:
                self._set_warmup_passthrough(False)

    async def on_stream_start(self):
        """Called when a new stream starts - enable loading overlay and queue warmup."""
        logger.info("Stream started, enabling loading overlay and scheduling warmup")
        try:
            self._reset_stop_event()
            self._frame_counter = 0
            self._loading_active = True
            self._warmup_done.clear()

            # Ensure pipeline/model are available
            if self.pipeline is None:
                await self.load_model()

            # Start text forwarder best-effort
            self._setup_text_monitoring()

            await self._start_warmup_sequence()
        except Exception as e:
            logger.error(f"on_stream_start failed: {e}")
            self._loading_active = False
            self._warmup_done.set()

    def _set_warmup_passthrough(self, enabled: bool) -> None:
        """Enable/disable passthrough during warmup (video only).

        When enabled, raw frames are passed through during warmup.
        When disabled, a loading overlay is rendered for video during warmup.
        """
        try:
            self._warmup_passthrough_enabled = bool(enabled)
            logger.info(
                "Warmup passthrough %s",
                "enabled" if self._warmup_passthrough_enabled else "disabled",
            )
        except Exception:
            logger.debug("Failed to set warmup passthrough flag", exc_info=True)

    def _schedule_warmup(self) -> None:
        """Schedule warmup in background if not already running."""
        try:
            if self._warmup_task and not self._warmup_task.done():
                logger.info("Warmup already in progress, skipping new warmup request")
                return

            self._frame_counter = 0
            self._loading_active = True
            self._warmup_done.clear()
            asyncio.get_running_loop().create_task(self._start_warmup_sequence())
            logger.info("Warmup scheduled in background")
        except Exception:
            logger.warning("Failed to schedule warmup", exc_info=True)

    async def process_video_async(self, frame: VideoFrame) -> VideoFrame:
        """Process video frame through ComfyStream Pipeline or emit a loading overlay during warmup."""
        try:
            if not self.pipeline:
                return frame

            if self._loading_active and not self._warmup_done.is_set():
                if self._warmup_passthrough_enabled:
                    return frame
                return self._build_loading_overlay_frame(frame)

            await self._ensure_runner_active()

            # Convert pytrickle VideoFrame to av.VideoFrame
            av_frame = frame.to_av_frame(frame.tensor)
            av_frame.pts = frame.timestamp
            av_frame.time_base = frame.time_base

            await self.pipeline.put_video_frame(av_frame)
            processed_av_frame = await self.pipeline.get_processed_video_frame()

            return VideoFrame.from_av_frame_with_timing(processed_av_frame, frame)

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return frame

    async def process_audio_async(self, frame: AudioFrame) -> List[AudioFrame]:
        """Process audio frame through ComfyStream Pipeline or passthrough."""
        try:
            if not self.pipeline:
                return [frame]
            if self._loading_active and not self._warmup_done.is_set():
                return [frame]
            # On first frame of an active stream, start/resume runner
            await self._ensure_runner_active()

            # Audio processing needed - use pipeline
            av_frame = frame.to_av_frame()
            await self.pipeline.put_audio_frame(av_frame)
            processed_av_frame = await self.pipeline.get_processed_audio_frame()
            processed_frame = AudioFrame.from_av_audio(processed_av_frame)
            return [processed_frame]

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return [frame]

    async def update_params(self, params: dict):
        """Update processing parameters."""
        if not self.pipeline:
            return

        # Handle list input - take first element
        if isinstance(params, list) and params:
            params = params[0]

        # Validate parameters using the centralized validation
        validated = ComfyStreamParamsUpdateRequest(**params).model_dump()
        logger.info(f"Parameter validation successful, keys: {list(validated.keys())}")

        # Process prompts if provided
        if "prompts" in validated and validated["prompts"]:
            await self._process_prompts(validated["prompts"])

        # Update pipeline dimensions
        if "width" in validated:
            self.pipeline.width = int(validated["width"])
        if "height" in validated:
            self.pipeline.height = int(validated["height"])

        # Schedule warmup if requested
        if validated.get("warmup", False):
            self._schedule_warmup()


    async def _process_prompts(self, prompts, *, skip_warmup: bool = False):
        """Process and set prompts in the pipeline."""
        try:
            converted = convert_prompt(prompts, return_dict=True)

            # Set prompts in pipeline
            await self.pipeline.set_prompts([converted])
            logger.info(f"Prompts set successfully: {list(prompts.keys())}")

            # Trigger loading overlay and warmup sequence for new prompts unless suppressed
            if not skip_warmup:
                try:
                    await self._start_warmup_sequence()
                except Exception:
                    logger.debug("Failed to start warmup sequence after prompt update", exc_info=True)

            # Update text monitoring based on workflow capabilities
            if self.pipeline.produces_text_output():
                self._setup_text_monitoring()
            else:
                await self._stop_text_forwarder()

        except Exception as e:
            logger.error(f"Failed to process prompts: {e}")

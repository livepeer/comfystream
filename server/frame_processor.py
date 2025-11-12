import asyncio
import logging
import os
from typing import List, Union

from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import AudioFrame, VideoFrame
from pytrickle.loading_config import LoadingConfig, LoadingMode
from pytrickle.stream_processor import VideoProcessingResult

from comfystream.pipeline import Pipeline
from comfystream.pipeline_state import PipelineState
from comfystream.utils import ComfyStreamParamsUpdateRequest, convert_prompt, get_default_workflow

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
        # Initialize parent with loading config for automatic overlay
        super().__init__(
            loading_config=LoadingConfig(
                mode=LoadingMode.OVERLAY,
                message="Loading workflow...",
                enabled=True,
                auto_timeout_seconds=1.5,
            )
        )

        self.pipeline = None
        self._load_params = load_params
        self._text_poll_interval = text_poll_interval
        self._stream_processor = None
        self._warmup_task = None
        self._text_forward_task = None
        self._background_tasks = []
        self._stop_event = asyncio.Event()
        self._first_frame_received = False

    def _workflow_has_video(self) -> bool:
        """Return True if current workflow is expected to produce video output."""
        if not self.pipeline:
            return False
        try:
            capabilities = self.pipeline.get_workflow_io_capabilities()
            return bool(capabilities.get("video", {}).get("output", False))
        except Exception:
            logger.debug("Unable to determine workflow video capability", exc_info=True)
            return False

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
                                            logger.debug(
                                                "Text send failed; stopping text forwarder"
                                            )
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

        # Stop the ComfyStream client's prompt execution
        if self.pipeline:
            logger.info("Stopping ComfyStream client prompt execution")
            try:
                await self.pipeline.stop_prompts(cleanup=True)
            except Exception as e:
                logger.error(f"Error stopping ComfyStream client: {e}")

        # Stop text forwarder
        await self._stop_text_forwarder()

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

        # Ensure loading overlay is turned off when stream stops
        self.set_loading_active(False, message="Stream stopped")

    def _reset_stop_event(self):
        """Reset the stop event for a new stream."""
        self._stop_event.clear()

    async def on_stream_start(self):
        """Handle stream start lifecycle events."""
        logger.info("Stream starting")
        self._reset_stop_event()
        self._first_frame_received = False

        if not self.pipeline:
            logger.debug("Stream start requested before pipeline initialization")
            return

        if not self.pipeline.state_manager.is_initialized():
            logger.info("Pipeline not initialized; waiting for prompts before streaming")
            return

        # Enable loading overlay for video workflows until first frame
        show_overlay = self._workflow_has_video()
        if show_overlay:
            self.set_loading_active(True, message="Initializing stream...")
        else:
            self.set_loading_active(False, message="Stream ready")

        try:
            if (
                self.pipeline.state != PipelineState.STREAMING
                and self.pipeline.state_manager.can_stream()
            ):
                await self.pipeline.start_streaming()

            if self.pipeline.produces_text_output():
                self._setup_text_monitoring()
            else:
                await self._stop_text_forwarder()
        except Exception:
            logger.exception("Failed during stream start", exc_info=True)
            if show_overlay:
                self.set_loading_active(False, message="Stream error")
            raise

    async def prepare_stream_loading(self, message: str = "Loading workflow..."):
        """Prime the loading overlay prior to receiving stream frames."""
        if not self.pipeline:
            return
        self._first_frame_received = False
        self.set_loading_active(True, message=message)

    async def load_model(self, **kwargs):
        """Load model and initialize the pipeline."""
        params = {**self._load_params, **kwargs}

        if self.pipeline is None:
            self.pipeline = Pipeline(
                width=int(params.get("width", 512)),
                height=int(params.get("height", 512)),
                cwd=params.get("workspace", os.getcwd()),
                disable_cuda_malloc=params.get("disable_cuda_malloc", True),
                gpu_only=params.get("gpu_only", True),
                preview_method=params.get("preview_method", "none"),
                comfyui_inference_log_level=params.get("comfyui_inference_log_level", "INFO"),
                logging_level=params.get("comfyui_inference_log_level", "INFO"),
                blacklist_custom_nodes=["ComfyUI-Manager"],
            )
            await self.pipeline.initialize()

    async def warmup(self):
        """Warm up the pipeline."""
        if not self.pipeline:
            logger.warning("Warmup requested before pipeline initialization")
            return

        logger.info("Running pipeline warmup...")
        try:
            capabilities = self.pipeline.get_workflow_io_capabilities()
            logger.info(f"Detected I/O capabilities: {capabilities}")

            await self.pipeline.warmup()

        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    def _schedule_warmup(self) -> None:
        """Schedule warmup in background if not already running."""
        try:
            if self._warmup_task and not self._warmup_task.done():
                logger.info("Warmup already in progress, skipping new warmup request")
                return

            self._warmup_task = asyncio.create_task(self.warmup())
            logger.info("Warmup scheduled in background")
        except Exception:
            logger.warning("Failed to schedule warmup", exc_info=True)

    async def process_video_async(
        self, frame: VideoFrame
    ) -> Union[VideoFrame, VideoProcessingResult]:
        """Process video frame through ComfyStream Pipeline.

        Returns VideoProcessingResult.WITHHELD to trigger pytrickle's automatic overlay when
        processed frames are not yet available.
        """
        try:
            if not self.pipeline:
                return frame

            # If pipeline ingestion is paused, withhold frame so pytrickle renders the overlay
            if not self.pipeline.is_ingest_enabled():
                self.set_loading_active(True)
                return VideoProcessingResult.WITHHELD

            # Convert pytrickle VideoFrame to av.VideoFrame
            av_frame = frame.to_av_frame(frame.tensor)
            av_frame.pts = frame.timestamp
            av_frame.time_base = frame.time_base

            # Process through pipeline
            await self.pipeline.put_video_frame(av_frame)

            # Try to get processed frame with short timeout
            try:
                processed_av_frame = await asyncio.wait_for(
                    self.pipeline.get_processed_video_frame(),
                    timeout=0.05,
                )
                processed_frame = VideoFrame.from_av_frame_with_timing(processed_av_frame, frame)

                # First frame received - disable loading overlay
                if not self._first_frame_received:
                    self._first_frame_received = True
                    self.set_loading_active(False, message="Workflow ready")

                return processed_frame

            except asyncio.TimeoutError:
                # No frame ready yet - return withheld sentinel to trigger overlay
                self.set_loading_active(True)
                return VideoProcessingResult.WITHHELD

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return frame

    async def process_audio_async(self, frame: AudioFrame) -> List[AudioFrame]:
        """Process audio frame through ComfyStream Pipeline or passthrough."""
        try:
            if not self.pipeline:
                return [frame]

            # If pipeline ingestion is paused, passthrough audio
            if not self.pipeline.is_ingest_enabled():
                frame.side_data.skipped = True
                frame.side_data.passthrough = True
                return [frame]

            # Audio processing - use pipeline
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

        if "show_loading" in validated:
            if validated["show_loading"]:
                message = validated.get("loading_message") or "Loading workflow..."
                self.set_loading_active(True, message=message)
            else:
                message = validated.get("loading_message") or "Workflow ready"
                self.set_loading_active(False, message=message)

    async def _process_prompts(self, prompts):
        """Process and set prompts in the pipeline."""
        if not self.pipeline:
            logger.warning("Prompt update requested before pipeline initialization")
            return
        try:
            converted = convert_prompt(prompts, return_dict=True)

            self.set_loading_active(True, message="Loading workflow...")

            capabilities = await self.pipeline.apply_prompts(
                [converted],
                skip_warmup=False,
            )

            video_capability = capabilities.get("video", {})
            has_video_output = bool(video_capability.get("output"))

            if has_video_output:
                self._first_frame_received = False
            else:
                self.set_loading_active(False, message="Workflow ready")

            if self.pipeline.state_manager.can_stream():
                await self.pipeline.start_streaming()

            logger.info(f"Prompts applied successfully: {list(prompts.keys())}")

            if self.pipeline.produces_text_output():
                self._setup_text_monitoring()
            else:
                await self._stop_text_forwarder()

        except Exception as e:
            logger.error(f"Failed to process prompts: {e}")
            self.set_loading_active(False, message="Workflow error")
            raise

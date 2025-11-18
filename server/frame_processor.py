import asyncio
import logging
import os
from fractions import Fraction
from typing import List

import numpy as np
import torch
from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import AudioFrame, VideoFrame

from comfystream.pipeline import Pipeline
from comfystream.utils import ComfyStreamParamsUpdateRequest, convert_prompt

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
        self._generative_video_task = None
        self._generative_audio_task = None
        self._generative_video_pts = 0
        self._generative_audio_pts = 0
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

    async def _stop_generative_video_forwarder(self) -> None:
        """Stop the background generative video task if running."""
        task = self._generative_video_task
        if task and not task.done():
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Error while awaiting generative video cancellation", exc_info=True)
        self._generative_video_task = None
        try:
            self._background_tasks.remove(task)
        except (ValueError, TypeError):
            pass

    async def _stop_generative_audio_forwarder(self) -> None:
        """Stop the background generative audio task if running."""
        task = self._generative_audio_task
        if task and not task.done():
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Error while awaiting generative audio cancellation", exc_info=True)
        self._generative_audio_task = None
        try:
            self._background_tasks.remove(task)
        except (ValueError, TypeError):
            pass

    async def _stop_generative_forwarders(self) -> None:
        """Stop all generative forwarder tasks."""
        await self._stop_generative_video_forwarder()
        await self._stop_generative_audio_forwarder()

    def _start_generative_video_forwarder(self) -> None:
        """Start the generative video forwarder if needed."""
        if self._generative_video_task and not self._generative_video_task.done():
            return
        if not self.pipeline or not self._stream_processor:
            return

        async def _generative_video_loop():
            logger.info("Starting generative video forwarder task")
            fps = getattr(self.pipeline, "frame_rate", None) or 30
            time_base = Fraction(1, int(fps))
            pts = self._generative_video_pts
            try:
                while not self._stop_event.is_set():
                    try:
                        out_tensor = await self.pipeline.client.get_video_output()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error(f"Failed to retrieve generative video output: {exc}")
                        await asyncio.sleep(0.1)
                        continue

                    if out_tensor is None:
                        await asyncio.sleep(0.01)
                        continue

                    processed_frame = self.pipeline.video_postprocess(out_tensor)
                    processed_frame.pts = pts
                    processed_frame.time_base = time_base

                    frame_np = processed_frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                    tensor = torch.from_numpy(frame_np)
                    video_frame = VideoFrame.from_tensor(tensor, timestamp=pts)
                    video_frame.time_base = time_base

                    success = await self._stream_processor.send_input_frame(video_frame)
                    if not success:
                        await asyncio.sleep(0.05)
                    pts += 1
                    self._generative_video_pts = pts
            except asyncio.CancelledError:
                logger.debug("Generative video forwarder cancelled")
                raise
            except Exception as exc:
                logger.error(f"Generative video forwarder encountered an error: {exc}")
            finally:
                logger.info("Generative video forwarder task exiting")

        self._generative_video_task = asyncio.create_task(_generative_video_loop())
        self._background_tasks.append(self._generative_video_task)

    def _start_generative_audio_forwarder(self) -> None:
        """Start the generative audio forwarder if needed."""
        if self._generative_audio_task and not self._generative_audio_task.done():
            return
        if not self.pipeline or not self._stream_processor:
            return

        async def _generative_audio_loop():
            logger.info("Starting generative audio forwarder task")
            sample_rate = 48000
            time_base = Fraction(1, sample_rate)
            pts = self._generative_audio_pts
            try:
                while not self._stop_event.is_set():
                    try:
                        out_audio = await self.pipeline.client.get_audio_output()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error(f"Failed to retrieve generative audio output: {exc}")
                        await asyncio.sleep(0.1)
                        continue

                    if out_audio is None:
                        await asyncio.sleep(0.01)
                        continue

                    processed_frame = self.audio_postprocess(out_audio)
                    processed_frame.pts = pts
                    processed_frame.time_base = time_base
                    processed_frame.sample_rate = sample_rate

                    audio_frame = AudioFrame.from_av_audio(processed_frame)
                    success = await self._stream_processor.send_input_frame(audio_frame)
                    if not success:
                        await asyncio.sleep(0.05)
                    pts += audio_frame.nb_samples
                    self._generative_audio_pts = pts
            except asyncio.CancelledError:
                logger.debug("Generative audio forwarder cancelled")
                raise
            except Exception as exc:
                logger.error(f"Generative audio forwarder encountered an error: {exc}")
            finally:
                logger.info("Generative audio forwarder task exiting")

        self._generative_audio_task = asyncio.create_task(_generative_audio_loop())
        self._background_tasks.append(self._generative_audio_task)

    async def _update_generative_forwarders(self) -> None:
        """Start or stop generative forwarders based on workflow capabilities."""
        if not self.pipeline or not self._stream_processor:
            return

        capabilities = self.pipeline.get_workflow_io_capabilities()
        video_only_output = capabilities["video"]["output"] and not capabilities["video"]["input"]
        audio_only_output = capabilities["audio"]["output"] and not capabilities["audio"]["input"]

        if video_only_output:
            self._start_generative_video_forwarder()
        else:
            await self._stop_generative_video_forwarder()

        if audio_only_output:
            self._start_generative_audio_forwarder()
        else:
            await self._stop_generative_audio_forwarder()

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

        # Stop generative forwarders
        await self._stop_generative_forwarders()

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

    def _reset_stop_event(self):
        """Reset the stop event for a new stream."""
        self._stop_event.clear()

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

    async def warmup(self):
        """Warm up the pipeline."""
        if not self.pipeline:
            logger.warning("Warmup requested before pipeline initialization")
            return

        logger.info("Running pipeline warmup...")
        try:
            capabilities = self.pipeline.get_workflow_io_capabilities()
            logger.info(f"Detected I/O capabilities: {capabilities}")

            if capabilities.get("video", {}).get("input") or capabilities.get("video", {}).get(
                "output"
            ):
                await self.pipeline.warm_video()

            if capabilities.get("audio", {}).get("input") or capabilities.get("audio", {}).get(
                "output"
            ):
                await self.pipeline.warm_audio()

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

    async def process_video_async(self, frame: VideoFrame) -> VideoFrame:
        """Process video frame through ComfyStream Pipeline."""
        try:
            # Convert pytrickle VideoFrame to av.VideoFrame
            av_frame = frame.to_av_frame(frame.tensor)
            av_frame.pts = frame.timestamp
            av_frame.time_base = frame.time_base

            # Process through pipeline
            await self.pipeline.put_video_frame(av_frame)
            processed_av_frame = await self.pipeline.get_processed_video_frame()

            # Convert back to pytrickle VideoFrame
            processed_frame = VideoFrame.from_av_frame_with_timing(processed_av_frame, frame)
            return processed_frame

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return frame

    async def process_audio_async(self, frame: AudioFrame) -> List[AudioFrame]:
        """Process audio frame through ComfyStream Pipeline or passthrough."""
        try:
            if not self.pipeline:
                return [frame]

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

    async def _process_prompts(self, prompts):
        """Process and set prompts in the pipeline."""
        try:
            converted = convert_prompt(prompts, return_dict=True)

            # Set prompts in pipeline
            await self.pipeline.set_prompts([converted])
            await self.pipeline.resume_prompts()
            logger.info(f"Prompts set successfully: {list(prompts.keys())}")

            # Update text monitoring based on workflow capabilities
            if self.pipeline.produces_text_output():
                self._setup_text_monitoring()
            else:
                await self._stop_text_forwarder()

            await self._update_generative_forwarders()

        except Exception as e:
            logger.error(f"Failed to process prompts: {e}")

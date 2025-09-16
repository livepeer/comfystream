import asyncio
import json
import logging
import os
from typing import List

import numpy as np
from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from comfystream.pipeline import Pipeline
from comfystream.utils import load_prompt_from_file, get_default_workflow, ComfyStreamParamsUpdateRequest

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
        super().__init__()

    def set_stream_processor(self, stream_processor):
        """Set reference to StreamProcessor for data publishing."""
        self._stream_processor = stream_processor
        logger.info("StreamProcessor reference set for text data publishing")
        
        # Set up text monitoring if pipeline is already loaded
        #if self.pipeline:
        #    self._setup_text_monitoring()
    
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
                width=int(params.get('width', 512)),
                height=int(params.get('height', 512)),
                cwd=params.get('workspace', os.getcwd()),
                disable_cuda_malloc=params.get('disable_cuda_malloc', True),
                gpu_only=params.get('gpu_only', True),
                preview_method=params.get('preview_method', 'none'),
                comfyui_inference_log_level=params.get('comfyui_inference_log_level'),
            )
        
        # Store warmup workflow for later use in update_params
        self._warmup_workflow = params.get('warmup_workflow')
        if self._warmup_workflow:
            try:
                workflow_data = load_prompt_from_file(self._warmup_workflow)
                logger.info(f"Loaded workflow data: {list(workflow_data.keys()) if workflow_data else 'None'}")
                logger.info(f"Node types in workflow: {[node.get('class_type') for node in workflow_data.values()] if workflow_data else 'None'}")
                self._warmup_workflow_data = workflow_data
            except Exception as e:
                logger.error(f"Failed to load workflow: {e}")
                self._warmup_workflow_data = None
        else:
            self._warmup_workflow_data = None
        
        # Set up text monitoring after pipeline is ready
        #self._setup_text_monitoring()
    
    async def _start_comfyui(self):
            
        try:
            # Use warmup workflow if available, otherwise default workflow
            if self._warmup_workflow_data:
                workflow_data = self._warmup_workflow_data
                logger.info("Starting ComfyUI with warmup workflow")
            else:
                workflow_data = get_default_workflow()
                logger.info("Starting ComfyUI with default workflow")
            
            # Set prompts to start ComfyUI
            await self.pipeline.set_prompts([workflow_data])
            logger.info("ComfyUI started successfully")

            # Start text monitoring if applicable
            # self._setup_text_monitoring()
            
            # Run warmup if we have warmup workflow
            if self._warmup_workflow_data:
                await self._run_warmup()
                logger.info("Warmup completed successfully")
                
        except Exception as e:
            logger.error(f"Failed to start ComfyUI: {e}")
            raise

    async def warmup(self):
        """Public warmup method that triggers pipeline warmup."""
        if not self.pipeline:
            logger.warning("Warmup requested before pipeline initialization")
            return
        
        logger.info("Running pipeline warmup...")
        """Run pipeline warmup."""
        try:
            capabilities = self.pipeline.get_workflow_io_capabilities()
            logger.info(f"Detected I/O capabilities for warmup: {capabilities}")
            
            # Warm video if there are video inputs or outputs
            if capabilities.get("video", {}).get("input") or capabilities.get("video", {}).get("output"):
                logger.info("Running video warmup...")
                await self.pipeline.warm_video()
                logger.info("Video warmup completed")
            
            # Warm audio if there are audio inputs or outputs  
            if capabilities.get("audio", {}).get("input") or capabilities.get("audio", {}).get("output"):
                logger.info("Running audio warmup...")
                await self.pipeline.warm_audio()
                logger.info("Audio warmup completed")
                
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
        
        try:
            # Detect sentinel early but evaluate truthiness after validation too
            requested_warmup = False
            try:
                requested_warmup = bool(params.get("warmup")) if isinstance(params, dict) else False
            except Exception:
                requested_warmup = False

            if isinstance(params, list) and params:
                params = params[0]
            
            logger.info(f"About to validate params: {type(params)}")
            validated = ComfyStreamParamsUpdateRequest(**params).model_dump()
            logger.info(f"Validation successful, validated keys: {list(validated.keys())}")
            
            
            # Preserve original prompt if directly provided as dict
            original_prompt_dict = None
            try:
                if isinstance(params, dict) and isinstance(params.get("prompts"), dict):
                    original_prompt_dict = params.get("prompts")
            except Exception:
                original_prompt_dict = None
            
            # Prefer raw params for prompts to avoid over-normalization, fallback to validated
            raw_prompts = None
            if isinstance(params, dict) and params.get("prompts") is not None:
                raw_prompts = params.get("prompts")
            elif "prompts" in validated and validated["prompts"]:
                raw_prompts = validated["prompts"]

            if raw_prompts is not None:
                # Normalize to a single prompt dict
                prompt_dict = None
                try:
                    # Accept Pydantic-style objects
                    if hasattr(raw_prompts, "model_dump"):
                        raw_prompts = raw_prompts.model_dump()
                    elif hasattr(raw_prompts, "dict") and callable(getattr(raw_prompts, "dict")):
                        raw_prompts = raw_prompts.dict()

                    if isinstance(raw_prompts, str):
                        parsed = json.loads(raw_prompts)
                        if hasattr(parsed, "model_dump"):
                            parsed = parsed.model_dump()
                        if isinstance(parsed, dict):
                            prompt_dict = parsed
                        elif isinstance(parsed, list):
                            prompt_dict = next((p for p in parsed if isinstance(p, dict)), None)
                    elif isinstance(raw_prompts, list):
                        # Prefer first dict entry, otherwise try to parse strings
                        prompt_dict = next((p for p in raw_prompts if isinstance(p, dict)), None)
                        if prompt_dict is None:
                            for item in raw_prompts:
                                if isinstance(item, str):
                                    try:
                                        candidate = json.loads(item)
                                        if isinstance(candidate, dict):
                                            prompt_dict = candidate
                                            break
                                    except Exception:
                                        continue
                    elif isinstance(raw_prompts, dict):
                        prompt_dict = raw_prompts
                except Exception as parse_e:
                    logger.warning(f"Failed to parse/normalize prompts: {parse_e}")

                if not isinstance(prompt_dict, dict):
                    # Fallback to original dict if available
                    if isinstance(original_prompt_dict, dict):
                        prompt_dict = original_prompt_dict
                    else:
                        logger.error("Prompts normalization failed: expected a single prompt dict")
                        prompt_dict = None

                if isinstance(prompt_dict, dict):
                    # Log pre-conversion keys/types
                    try:
                        node_types = [node.get('class_type') for node in prompt_dict.values()]
                    except Exception:
                        node_types = []
                    logger.info(f"Setting prompts in pipeline: {list(prompt_dict.keys())}")
                    logger.info(f"Node types before conversion: {node_types}")

                    # Convert to comfy format dict
                    from comfystream.utils import convert_prompt
                    try:
                        converted = convert_prompt(prompt_dict, return_dict=True)
                    except Exception as conv_e:
                        logger.error(f"Prompt conversion failed: {conv_e}")
                        converted = prompt_dict

                    await self.pipeline.set_prompts([converted])
                    # Start or stop text monitoring based on updated workflow modalities
                    try:
                        if bool(self.pipeline.produces_text_output()):
                            self._setup_text_monitoring()
                        else:
                            await self._stop_text_forwarder()
                    except Exception:
                        # Best-effort: if capability detection fails, do nothing here
                        logger.debug("Unable to determine text output capability after prompts update", exc_info=True)
            
            if "width" in validated:
                self.pipeline.width = int(validated["width"])
            if "height" in validated:
                self.pipeline.height = int(validated["height"])
            
            # Schedule warmup if requested via sentinel flag
            if requested_warmup or bool(validated.get("warmup", False)):
                self._schedule_warmup()
                # After warmup completes, start text monitoring exactly once here
                async def _start_monitoring_after_warmup():
                    try:
                        task = self._warmup_task
                        if task:
                            try:
                                await task
                            except asyncio.CancelledError:
                                return
                        # Start monitoring only after warmup
                        self._setup_text_monitoring()
                    except Exception:
                        logger.warning("Failed to start text monitoring after warmup", exc_info=True)

                try:
                    follow_up_task = asyncio.create_task(_start_monitoring_after_warmup())
                    self._background_tasks.append(follow_up_task)
                except Exception:
                    logger.debug("Unable to schedule post-warmup text monitoring task", exc_info=True)
                
        except Exception as e:
            logger.error(f"Parameter update failed: {e}")
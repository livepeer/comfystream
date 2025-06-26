import asyncio
import logging

logger = logging.getLogger(__name__)

async def _process_frame_batch(batch, pipeline, output_queue, metrics):
    """Process a batch of frames through pipeline with ComfyUI blocking protection"""
    try:
        for frame_item in batch:
            frame = frame_item['frame']
            original_pts = frame_item['original_pts']
            original_time_base = frame_item['original_time_base']
            segment_id = frame_item['segment_id']
            expected_frames = frame_item.get('expected_frames_in_segment', None)
            
            try:
                # Process through pipeline with patient timeout to allow ComfyUI to work
                await asyncio.wait_for(pipeline.put_video_frame(frame), timeout=2.0)  # More time for queuing
                processed_frame = await asyncio.wait_for(pipeline.get_processed_video_frame(), timeout=10.0)  # Much more time for processing
                
                # Restore timing
                processed_frame.pts = original_pts
                processed_frame.time_base = original_time_base
                
                # Queue with flow info including expected frame count
                processed_with_timing = {
                    'frame': processed_frame,
                    'segment_id': segment_id,
                    'processing_batch': True,
                    'expected_frames_in_segment': expected_frames
                }
                
                await output_queue.put(processed_with_timing)
                metrics['processed_frames'] += 1
                metrics['processed_queue_depth'] = output_queue.qsize()
                
            except asyncio.TimeoutError:
                # ComfyUI is severely blocked (>10s per frame) - this indicates a real problem
                logger.error(f"ComfyUI severely blocked (>10s timeout), using original frame for segment {segment_id}")
                
                # Try to clear the pipeline state in case it's stuck
                try:
                    # Attempt to flush the pipeline (non-blocking)
                    while True:
                        try:
                            await asyncio.wait_for(pipeline.get_processed_video_frame(), timeout=0.1)
                        except:
                            break
                except:
                    pass
                
                # Use original frame as fallback only after attempting pipeline recovery
                frame.pts = original_pts
                frame.time_base = original_time_base
                
                processed_with_timing = {
                    'frame': frame,  # Use original frame
                    'segment_id': segment_id,
                    'processing_batch': True,
                    'bypass_comfyui': True,  # Mark as bypassed
                    'expected_frames_in_segment': expected_frames
                }
                
                await output_queue.put(processed_with_timing)
                metrics['processed_frames'] += 1
                metrics['bypassed_frames'] = metrics.get('bypassed_frames', 0) + 1
                metrics['processed_queue_depth'] = output_queue.qsize()
                
                # Add a brief pause to prevent cascade failures
                await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error processing frame batch: {e}")

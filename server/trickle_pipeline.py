import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from comfystream.server.trickle import TrickleSegmentEncoder, TrickleSubscriber, TrickleStreamDecoder
from process import _process_frame_batch

logger = logging.getLogger(__name__)

@dataclass
class StreamManifest:
    """Represents a streaming session with manifest ID for tracking"""
    manifest_id: str
    input_stream_url: str  # Input stream URL for trickle subscription
    output_stream_url: str  # Output stream URL for trickle publishing
    created_at: datetime
    status: str  # 'starting', 'active', 'stopping', 'stopped'
    pipeline: Optional[Any] = None
    publisher_task: Optional[asyncio.Task] = None
    subscriber_task: Optional[asyncio.Task] = None
    frame_processor_task: Optional[asyncio.Task] = None
    frame_queue: Optional[asyncio.Queue] = None
    metadata: Optional[Dict[str, Any]] = None
    encoder: Optional[TrickleSegmentEncoder] = None  # Persistent encoder for timestamp continuity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "manifest_id": self.manifest_id,
            "input_stream_url": self.input_stream_url,
            "output_stream_url": self.output_stream_url,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "metadata": self.metadata or {}
        }

async def _read_complete_segment(segment_reader) -> bytes:
    """Read the complete trickle segment data"""
    try:
        complete_data = b""
        while True:
            chunk = await segment_reader.read(8192)
            if not chunk:
                break
            complete_data += chunk
        return complete_data
    except Exception as e:
        logger.error(f"Error reading segment: {e}")
        return b""

async def _run_trickle_pipeline_airunner_style(input_url: str, output_url: str, pipeline, frame_queue, width: int, height: int, stream_manifest: StreamManifest):
    """Async decoupled trickle pipeline with flow control for smooth output"""
    try:
        import av
        logger.info("Starting async decoupled trickle pipeline with flow control")
        
        # Large queues for maximum flexibility and buffering
        input_segment_queue = asyncio.Queue(maxsize=50)      # ~10+ input segments
        decoded_frame_queue = asyncio.Queue(maxsize=1000)    # ~13+ segments worth - plenty of ComfyUI buffer
        processed_frame_queue = asyncio.Queue(maxsize=800)   # ~10+ segments worth for smooth output
        output_control_queue = asyncio.Queue(maxsize=200)    # Large output buffering
        
        stream_decoder = TrickleStreamDecoder(target_width=width, target_height=height)
        
        # Metrics for monitoring flow
        flow_metrics = {
            'input_segments': 0,
            'decoded_frames': 0, 
            'processed_frames': 0,
            'published_segments': 0,
            'input_queue_depth': 0,
            'decoded_queue_depth': 0,
            'processed_queue_depth': 0,
            'output_queue_depth': 0
        }
        
        # Task 1: Async Trickle Subscriber (continuous input buffering)
        async def trickle_subscriber_task():
            """Continuously fetch and buffer trickle segments for smooth flow"""
            try:
                logger.info("Starting async trickle subscriber with enhanced buffering")
                async with TrickleSubscriber(input_url) as subscriber:
                    segment_count = 0
                    consecutive_empty = 0
                    max_consecutive_empty = 50  # Allow more retries
                    
                    while stream_manifest.status == 'active':
                        try:
                            # Fetch segment with backoff on empty
                            current_segment = await subscriber.next()
                            if current_segment is None:
                                consecutive_empty += 1
                                if consecutive_empty >= max_consecutive_empty:
                                    logger.info("No more segments available, ending subscriber")
                                    break
                                # Progressive backoff for empty segments
                                sleep_time = min(0.1 * consecutive_empty, 2.0)
                                await asyncio.sleep(sleep_time)
                                continue
                            
                            consecutive_empty = 0  # Reset counter on successful fetch
                            segment_count += 1
                            
                            # Read segment data
                            segment_data = await _read_complete_segment(current_segment)
                            if segment_data:
                                # Queue segment with priority handling
                                try:
                                    await asyncio.wait_for(
                                        input_segment_queue.put((segment_count, segment_data)), 
                                        timeout=0.5  # Longer timeout for buffering
                                    )
                                    flow_metrics['input_segments'] += 1
                                    flow_metrics['input_queue_depth'] = input_segment_queue.qsize()
                                    
                                    if segment_count % 10 == 0:
                                        logger.debug(f"Buffered segment {segment_count} (queue depth: {flow_metrics['input_queue_depth']})")
                                        
                                except asyncio.TimeoutError:
                                    # If queue is full, wait longer rather than dropping
                                    logger.warning(f"Input queue full, waiting for space...")
                                    await input_segment_queue.put((segment_count, segment_data))
                            
                            # Close segment
                            if hasattr(current_segment, 'close'):
                                try:
                                    await current_segment.close()
                                except:
                                    pass
                                    
                        except Exception as e:
                            logger.error(f"Error in subscriber: {e}")
                            await asyncio.sleep(0.2)
                            
                logger.info(f"Trickle subscriber finished: {segment_count} segments buffered")
                            
            except Exception as e:
                logger.error(f"Trickle subscriber task error: {e}")
            finally:
                # Signal end of input
                await input_segment_queue.put(None)
        
        # Task 2: Async Decoder (continuous frame decoding)
        async def decoder_task():
            """Decode segments into frames maintaining large buffer"""
            try:
                logger.info("Starting async decoder with enhanced buffering")
                total_frames = 0
                
                while True:
                    try:
                        # Get segment from queue with longer timeout
                        segment_item = await asyncio.wait_for(
                            input_segment_queue.get(), 
                            timeout=5.0  # Longer timeout for flow
                        )
                        
                        if segment_item is None:
                            break  # End of input
                        
                        segment_count, segment_data = segment_item
                        flow_metrics['input_queue_depth'] = input_segment_queue.qsize()
                        
                        # Decode segment
                        decoded_frames = stream_decoder.process_segment(segment_data)
                        
                        if decoded_frames:
                            # Queue each frame with preserved timing and segment frame count
                            total_frames_in_segment = len(decoded_frames)
                            for i, frame in enumerate(decoded_frames):
                                frame_with_timing = {
                                    'frame': frame,
                                    'original_pts': frame.pts,
                                    'original_time_base': frame.time_base,
                                    'segment_id': segment_count,
                                    'frame_index_in_segment': i,
                                    'expected_frames_in_segment': total_frames_in_segment  # Add expected count
                                }
                                
                                try:
                                    # Large queue should accommodate ComfyUI processing variations
                                    queue_size = decoded_frame_queue.qsize()
                                    if queue_size > 900:  # High usage warning for 1000-frame queue
                                        logger.warning(f"Decoded frame queue high usage ({queue_size}/1000)")
                                    
                                    await asyncio.wait_for(
                                        decoded_frame_queue.put(frame_with_timing),
                                        timeout=2.0  # Longer timeout with large queue
                                    )
                                    total_frames += 1
                                    flow_metrics['decoded_frames'] += 1
                                    flow_metrics['decoded_queue_depth'] = decoded_frame_queue.qsize()
                                    
                                except asyncio.TimeoutError:
                                    # With large queue, this should be rare - wait rather than drop
                                    logger.warning(f"Decoded frame queue full, waiting... (queue: {decoded_frame_queue.qsize()}/1000)")
                                    await decoded_frame_queue.put(frame_with_timing)
                                    total_frames += 1
                            
                            if segment_count % 5 == 0:
                                logger.debug(f"Decoded segment {segment_count}: {len(decoded_frames)} frames (total: {total_frames}, queue: {flow_metrics['decoded_queue_depth']})")
                        
                    except asyncio.TimeoutError:
                        if stream_manifest.status != 'active':
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Error in decoder: {e}")
                        continue
                
                logger.info(f"Decoder finished: {total_frames} frames decoded")
                
            except Exception as e:
                logger.error(f"Decoder task error: {e}")
            finally:
                await decoded_frame_queue.put(None)
        
        # Task 3: Async Pipeline Processor (managed AI processing)
        async def pipeline_processor_task():
            """Process frames through ComfyUI with flow management and queue monitoring"""
            try:
                logger.info("Starting async pipeline processor with ComfyUI queue monitoring")
                processed_count = 0
                processing_batch_size = 3  # Smaller batches to prevent blocking
                current_batch = []
                last_queue_warning = 0
                
                while True:
                    try:
                        # Monitor queue sizes to detect ComfyUI blocking
                        decoded_queue_size = decoded_frame_queue.qsize()
                        processed_queue_size = processed_frame_queue.qsize()
                        
                        # Monitor large queues (warn at higher thresholds)
                        if decoded_queue_size > 800 and time.time() - last_queue_warning > 10.0:
                            logger.warning(f"High queue usage: decoded_queue={decoded_queue_size}/1000, "
                                         f"processed_queue={processed_queue_size}/800")
                            last_queue_warning = time.time()
                        
                        # Get frame from decoder queue with very short timeout to prevent blocking
                        frame_item = await asyncio.wait_for(
                            decoded_frame_queue.get(),
                            timeout=0.1  # Very short timeout to detect blocking immediately
                        )
                        
                        if frame_item is None:
                            # Process any remaining batch
                            if current_batch:
                                await _process_frame_batch(current_batch, pipeline, processed_frame_queue, flow_metrics)
                                processed_count += len(current_batch)
                            break
                        
                        flow_metrics['decoded_queue_depth'] = decoded_frame_queue.qsize()
                        current_batch.append(frame_item)
                        
                        # Process in smaller batches to maintain flow
                        if len(current_batch) >= processing_batch_size:
                            await _process_frame_batch(current_batch, pipeline, processed_frame_queue, flow_metrics)
                            processed_count += len(current_batch)
                            current_batch = []
                            
                            # Brief pause to allow other tasks to work
                            await asyncio.sleep(0.01)
                        
                        # Let queue fill naturally - no emergency processing that adds chaos
                        
                        # Also process batch if we have any frames and queue is not completely full
                        elif len(current_batch) > 0 and decoded_frame_queue.qsize() < 100:
                            await _process_frame_batch(current_batch, pipeline, processed_frame_queue, flow_metrics)
                            processed_count += len(current_batch)
                            current_batch = []
                        
                    except asyncio.TimeoutError:
                        # PATIENT TIMEOUT PROCESSING: Only intervene when ComfyUI is truly blocked
                        decoded_queue_size = decoded_frame_queue.qsize()
                        processed_queue_size = processed_frame_queue.qsize()
                        
                        # NATURAL QUEUE MANAGEMENT: Let ComfyUI work at its own pace
                        # Log queue status for monitoring but don't intervene with aggressive dropping
                        if decoded_queue_size > 800:
                            logger.debug(f"ComfyUI processing: decoded={decoded_queue_size}/1000, processed={processed_queue_size}/800")
                        
                        # Process current batch on timeout to maintain flow
                        if current_batch:
                            await _process_frame_batch(current_batch, pipeline, processed_frame_queue, flow_metrics)
                            processed_count += len(current_batch)
                            current_batch = []
                        
                        if stream_manifest.status != 'active':
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        current_batch = []  # Reset batch on error
                        continue
                
                logger.info(f"Pipeline processor finished: {processed_count} frames processed")
                
            except Exception as e:
                logger.error(f"Pipeline processor task error: {e}")
            finally:
                await processed_frame_queue.put(None)
        
        # Task 4: Segment-Aware Output Controller (maintains input/output correspondence)
        async def output_flow_controller():
            """Control output with perfect segment correspondence and buffer warmup"""
            try:
                logger.info("Starting segment-aware output controller with buffer warmup strategy")
                
                # Track segments by input segment ID to maintain correspondence
                active_segments = {}  # segment_id -> {'frames': [], 'published': bool, 'start_time': float}
                completed_segments = []
                
                # No buffer warmup needed with large queues - start publishing immediately
                expected_frames_per_segment = 72  # 24 FPS * 3 seconds
                warmup_complete = True  # Always ready to publish with large queues
                
                logger.info("No buffer warmup delay - starting immediate publishing with large queue buffers")
                
                while True:
                    try:
                        # Get processed frame with metadata (very short timeout for aggressive flow)
                        frame_item = await asyncio.wait_for(
                            processed_frame_queue.get(),
                            timeout=1.0  # Very short timeout for maximum responsiveness
                        )
                        
                        if frame_item is None:
                            # Publish any remaining segments in order (only unpublished ones)
                            for segment_id in sorted(active_segments.keys()):
                                segment_info = active_segments[segment_id]
                                if segment_info['frames'] and not segment_info.get('published', False):
                                    await _segment_aware_publish(
                                        segment_info['frames'], stream_manifest.encoder,
                                        output_control_queue, segment_id
                                    )
                                    segment_info['published'] = True
                                    logger.info(f"Final segment {segment_id}: {len(segment_info['frames'])} frames")
                            break
                        
                        frame = frame_item['frame']
                        input_segment_id = frame_item['segment_id']
                        flow_metrics['processed_queue_depth'] = processed_frame_queue.qsize()
                        
                        # Initialize segment tracking if new
                        if input_segment_id not in active_segments:
                            active_segments[input_segment_id] = {
                                'frames': [],
                                'published': False,  # Track if this segment has been published
                                'start_time': time.time()  # Track when segment started for timeout publishing
                            }
                            logger.debug(f"Started tracking segment {input_segment_id}")
                        
                        # Skip if already published to prevent duplicates
                        if active_segments[input_segment_id].get('published', False):
                            logger.debug(f"Skipping frame for already published segment {input_segment_id}")
                            continue
                        
                        # Add frame to the corresponding segment
                        active_segments[input_segment_id]['frames'].append(frame)
                        current_count = len(active_segments[input_segment_id]['frames'])
                        
                        # Track expected frame count from decoder metadata
                        expected_frame_count = frame_item.get('expected_frames_in_segment', None)
                        
                        # With large queues, no warmup delay needed - always ready to publish
                        total_buffered_frames = processed_frame_queue.qsize()
                        
                        # ADAPTIVE STREAMING OPTIMIZED: Generous tolerance for natural frame variation
                        # Expected: ~72 frames per segment at 24 FPS * 3 seconds
                        expected_frames_24fps = 72
                        tolerance = 0.5  # 50% tolerance for natural frame count variation
                        min_frames = max(36, int(expected_frames_24fps * (1 - tolerance)))  # ~36 frames minimum (1.5 seconds)
                        max_frames = int(expected_frames_24fps * (1 + tolerance))  # ~108 frames maximum (4.5 seconds)
                        
                        segment_ready = False
                        if expected_frame_count is not None:
                            # CRITICAL: Use decoder's exact frame count - this maintains input stream correspondence
                            segment_ready = (current_count >= expected_frame_count)
                            if segment_ready:
                                logger.debug(f"Segment {input_segment_id} ready (EXACT INPUT MATCH): {current_count}/{expected_frame_count} frames")
                        else:
                            # NO TIMEOUT PUBLISHING - wait for input stream to determine natural segment boundaries
                            # This prevents PTS discontinuities and timing gaps
                            segment_ready = False
                            logger.debug(f"Segment {input_segment_id} waiting for input stream boundary: {current_count} frames collected")
                        
                        # NATURAL ACCUMULATION: Let segments reach their natural size
                        # Remove emergency publishing - trust the input stream boundaries
                        
                        # REMOVE TIMEOUT PUBLISHING - it breaks input stream correspondence and PTS tracking
                        # The input stream decoder should provide exact frame counts, timeout interferes with this
                        
                        # With large queues (800+ frames), no buffer maintenance delays needed
                        
                        if segment_ready and current_count > 0 and not active_segments[input_segment_id].get('published', False):
                            segment_frames = active_segments[input_segment_id]['frames']
                            await _segment_aware_publish(
                                segment_frames, stream_manifest.encoder,
                                output_control_queue, input_segment_id
                            )
                            
                            # Mark as published to prevent duplicates
                            active_segments[input_segment_id]['published'] = True
                            
                            logger.info(f"Published segment {input_segment_id}: {len(segment_frames)} frames "
                                      f"(buffer: {processed_frame_queue.qsize()} frames remaining)")
                            completed_segments.append(input_segment_id)
                        
                    except asyncio.TimeoutError:
                        # WAIT FOR INPUT STREAM BOUNDARIES: No timeout publishing that breaks correspondence
                        # Only publish segments that have completed naturally from input stream
                        
                        for segment_id in list(active_segments.keys()):
                            segment_info = active_segments[segment_id]
                            frame_count = len(segment_info['frames'])
                            
                            # PATIENT WAITING: Trust the input stream to provide boundaries
                            # No emergency publishing - let segments complete naturally
                            logger.debug(f"Waiting for input boundary: segment {segment_id} has {frame_count} frames")
                        
                        if stream_manifest.status != 'active':
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Error in segment-aware controller: {e}")
                        continue
                
                logger.info(f"Segment-aware controller finished: {len(completed_segments)} unique segments with perfect correspondence")
                
            except Exception as e:
                logger.error(f"Segment-aware controller error: {e}")
            finally:
                await output_control_queue.put(None)
        
        # Task 5: Adaptive Buffered Publisher (builds segment buffer and adapts to ComfyUI throughput)
        async def buffered_publisher_task():
            """Adaptive buffered publishing with throughput monitoring and FPS adaptation"""
            try:
                logger.info("Starting adaptive buffered publisher with throughput monitoring")
                
                segment_buffer = {}  # segment_id -> segment_data (ready segments)
                published_segments = set()  # Track published segment IDs to prevent duplicates
                next_publish_id = None  # Next segment ID to publish (sequential)
                buffer_target = 1  # Target buffer size (3 seconds ahead)
                buffer_warmup = 0  # No warmup - immediate publishing
                published_count = 0
                playback_started = False
                
                # Adaptive throughput monitoring
                segment_times = []  # Track processing times for throughput calculation
                adaptive_fps = 24.0  # Start with target FPS, adapt based on actual throughput
                last_throughput_check = time.time()
                throughput_check_interval = 15.0  # Check every 15 seconds
                
                logger.info(f"Adaptive buffer strategy: warmup={buffer_warmup} segments, target={buffer_target} segments, initial_fps={adaptive_fps}")
                
                while True:
                    try:
                        # Get completed segment from output controller
                        segment_item = await asyncio.wait_for(
                            output_control_queue.get(),
                            timeout=2.0
                        )
                        
                        if segment_item is None:
                            # Publish all remaining buffered segments in order
                            if segment_buffer:
                                logger.info(f"Stream ending: publishing {len(segment_buffer)} remaining buffered segments")
                                for segment_id in sorted(segment_buffer.keys()):
                                    if segment_id not in published_segments:
                                        await frame_queue.put((segment_id, segment_buffer[segment_id]))
                                        published_segments.add(segment_id)
                                        published_count += 1
                            break
                        
                        # Handle tuple format (segment_id, segment_data)
                        if isinstance(segment_item, tuple) and len(segment_item) == 2:
                            segment_id, segment_data = segment_item
                            
                            # Skip duplicates
                            if segment_id in published_segments or segment_id in segment_buffer:
                                logger.debug(f"Skipping duplicate segment {segment_id}")
                                continue
                            
                            # Add to buffer with timing tracking
                            segment_buffer[segment_id] = segment_data
                            current_time = time.time()
                            segment_times.append(current_time)
                            
                            # Set the starting point for sequential publishing
                            if next_publish_id is None:
                                next_publish_id = segment_id
                                logger.info(f"Starting segment sequence at ID {next_publish_id}")
                            
                            buffer_size = len(segment_buffer)
                            logger.debug(f"Buffered segment {segment_id} (buffer: {buffer_size}/{buffer_target} segments)")
                            
                            # ADAPTIVE THROUGHPUT MONITORING: Check if ComfyUI throughput requires FPS adaptation
                            if current_time - last_throughput_check > throughput_check_interval and len(segment_times) >= 5:
                                # Calculate actual segment processing rate over last period
                                recent_segments = [t for t in segment_times if t > last_throughput_check]
                                if len(recent_segments) >= 3:
                                    time_span = recent_segments[-1] - recent_segments[0]
                                    segments_per_second = (len(recent_segments) - 1) / time_span if time_span > 0 else 0
                                    
                                    # Each segment should be 3 seconds at target FPS
                                    # If we're getting segments slower, ComfyUI can't keep up with target FPS
                                    expected_segments_per_second = 1.0 / 3.0  # 1 segment every 3 seconds at full speed
                                    throughput_ratio = segments_per_second / expected_segments_per_second if expected_segments_per_second > 0 else 1.0
                                    
                                    # Calculate what FPS ComfyUI can actually sustain
                                    sustainable_fps = adaptive_fps * throughput_ratio
                                    
                                    # Adapt if ComfyUI is significantly slower than target
                                    if throughput_ratio < 0.8 and sustainable_fps < adaptive_fps:
                                        old_fps = adaptive_fps
                                        adaptive_fps = max(12.0, sustainable_fps)  # Never go below 12 FPS
                                        
                                        # Adjust buffer strategy for lower throughput
                                        if adaptive_fps < 20:
                                            buffer_target = 2  # Slightly larger buffer for slower throughput
                                            buffer_warmup = 1
                                        
                                        logger.warning(f"🔄 THROUGHPUT ADAPTATION: ComfyUI throughput {throughput_ratio:.2f}x target")
                                        logger.warning(f"📉 FPS ADAPTATION: {old_fps:.1f} → {adaptive_fps:.1f} FPS (buffer: {buffer_target} segments)")
                                    
                                    elif throughput_ratio > 1.1 and sustainable_fps > adaptive_fps:
                                        # ComfyUI is faster than expected, can increase FPS
                                        old_fps = adaptive_fps
                                        adaptive_fps = min(24.0, sustainable_fps)  # Cap at original target
                                        
                                        logger.info(f"📈 FPS IMPROVEMENT: {old_fps:.1f} → {adaptive_fps:.1f} FPS")
                                    
                                    last_throughput_check = current_time
                                    
                                    # Clean old timing data
                                    cutoff_time = current_time - 60.0  # Keep last 60 seconds
                                    segment_times = [t for t in segment_times if t > cutoff_time]
                            
                            # BUFFER WARMUP: Start immediately or when minimum buffer reached
                            if not playback_started and (buffer_warmup == 0 or buffer_size >= buffer_warmup):
                                playback_started = True
                                if buffer_warmup == 0:
                                    logger.info(f"🚀 IMMEDIATE PLAYBACK: Starting with no warmup delay")
                                else:
                                    logger.info(f"🚀 PLAYBACK STARTING: Buffer warmed up with {buffer_size} segments")
                            
                            # BUFFERED PUBLISHING: Maintain target buffer size
                            while playback_started and next_publish_id in segment_buffer and len(segment_buffer) > buffer_target:
                                # Publish the next sequential segment
                                segment_data_to_publish = segment_buffer.pop(next_publish_id)
                                published_segments.add(next_publish_id)
                                
                                await frame_queue.put((next_publish_id, segment_data_to_publish))
                                published_count += 1
                                
                                remaining_buffer = len(segment_buffer)
                                logger.info(f"📺 PUBLISHED segment {next_publish_id} (buffer: {remaining_buffer}/{buffer_target} segments remaining)")
                                
                                # Move to next segment in sequence
                                next_publish_id += 1
                                
                                # Brief pause to prevent overwhelming the publisher
                                await asyncio.sleep(0.1)
                            
                            # CONTINUOUS PUBLISHING: Once warmed up, publish ready segments to maintain buffer
                            if playback_started and next_publish_id in segment_buffer:
                                # Always publish if we have the next sequential segment ready
                                segment_data_to_publish = segment_buffer.pop(next_publish_id)
                                published_segments.add(next_publish_id)
                                
                                await frame_queue.put((next_publish_id, segment_data_to_publish))
                                published_count += 1
                                
                                remaining_buffer = len(segment_buffer)
                                logger.debug(f"📺 CONTINUOUS: Published segment {next_publish_id} (buffer: {remaining_buffer} segments)")
                                next_publish_id += 1
                        
                        else:
                            # Fallback for old format (should not happen with our setup)
                            await frame_queue.put(segment_item)
                            published_count += 1
                            logger.debug(f"Published segment {published_count} (fallback format)")
                        
                        flow_metrics['published_segments'] = published_count
                        flow_metrics['segment_buffer_size'] = len(segment_buffer)
                        flow_metrics['output_queue_depth'] = output_control_queue.qsize()
                        
                        # Log buffer status periodically
                        if published_count % 5 == 0 and published_count > 0:
                            logger.info(f"📊 Buffer status: {published_count} published, {len(segment_buffer)} buffered, next_id={next_publish_id}")
                        
                    except asyncio.TimeoutError:
                        # During timeout, check if we can publish any ready segments
                        if playback_started and next_publish_id in segment_buffer:
                            segment_data_to_publish = segment_buffer.pop(next_publish_id)
                            published_segments.add(next_publish_id)
                            
                            await frame_queue.put((next_publish_id, segment_data_to_publish))
                            published_count += 1
                            
                            logger.debug(f"📺 TIMEOUT: Published segment {next_publish_id} (buffer: {len(segment_buffer)} segments)")
                            next_publish_id += 1 # type: ignore #
                        
                        if stream_manifest.status != 'active':
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Error in buffered publisher: {e}")
                        continue
                
                logger.info(f"Buffered publisher finished: {published_count} segments published with buffer strategy")
                
            except Exception as e:
                logger.error(f"Buffered publisher error: {e}")
            finally:
                await frame_queue.put(None)
        
        # Start all async tasks with enhanced flow control
        logger.info("Starting all async tasks with flow control")
        
        tasks = await asyncio.gather(
            asyncio.create_task(trickle_subscriber_task()),
            asyncio.create_task(decoder_task()),
            asyncio.create_task(pipeline_processor_task()),
            asyncio.create_task(output_flow_controller()),
            asyncio.create_task(buffered_publisher_task()),
            return_exceptions=True
        )
        
        # Log task completion and final metrics
        task_names = ["subscriber", "decoder", "pipeline", "flow_controller", "publisher"]
        for i, result in enumerate(tasks):
            if isinstance(result, Exception):
                logger.error(f"Task {task_names[i]} failed: {result}")
            else:
                logger.info(f"Task {task_names[i]} completed successfully")
        
        logger.info(f"Final flow metrics: {flow_metrics}")
        logger.info("Async flow-controlled trickle pipeline finished")
        
    except Exception as e:
        logger.error(f"Flow-controlled pipeline error: {e}")

async def _segment_aware_publish(frames, encoder, output_queue, segment_id):
    """Publish segment with segment ID preservation"""
    try:
        if not frames or not encoder:
            return
        
        # Encode with preserved timing
        segment_data = encoder.encode_frames_batch(frames, segment_id)
        
        if segment_data:
            # Pass segment data with ID for correspondence
            await output_queue.put((segment_id, segment_data))
            logger.debug(f"Segment-aware publish: segment {segment_id}, {len(frames)} frames, {len(segment_data)} bytes")
        else:
            logger.error(f"Failed to encode segment {segment_id}")
            
    except Exception as e:
        logger.error(f"Error in segment-aware publish {segment_id}: {e}")

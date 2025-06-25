import time
import asyncio
import logging
import os
import threading
from typing import Callable, Any, Optional

from .trickle_publisher import TricklePublisher
from .encoder import TrickleMetadataExtractor

logger = logging.getLogger(__name__)

MAX_ENCODER_RETRIES = 3
ENCODER_RETRY_RESET_SECONDS = 120  # reset retry counter after 2 minutes

async def run_publish(publish_url: str, frame_generator: Callable, get_metadata: Callable, monitoring_callback: Optional[Callable] = None):
    """
    Run the publishing pipeline for trickle streaming.
    
    Args:
        publish_url: URL to publish the stream to
        frame_generator: Generator function that yields video frames
        get_metadata: Function to get stream metadata
        monitoring_callback: Optional callback for monitoring events
    """
    first_segment = True
    publisher = None
    
    try:
        publisher = TricklePublisher(url=publish_url, mime_type="video/mp2t")
        
        loop = asyncio.get_running_loop()
        
        async def segment_callback(pipe_file, pipe_name):
            nonlocal first_segment
            # trickle publish a segment with the contents of `pipe_file`
            async with await publisher.next() as segment:
                # convert pipe_fd into an asyncio friendly StreamReader
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                transport, _ = await loop.connect_read_pipe(lambda: protocol, pipe_file)
                
                while True:
                    sz = 32 * 1024  # read in chunks of 32KB
                    data = await reader.read(sz)
                    if not data:
                        break
                    await segment.write(data)
                    
                    if first_segment and monitoring_callback:
                        first_segment = False
                        await monitoring_callback({
                            "type": "runner_send_first_processed_segment",
                            "timestamp": int(time.time() * 1000)
                        }, queue_event_type="stream_trace")
                        
                transport.close()

        def sync_callback(pipe_reader, pipe_writer, pipe_name):
            def do_schedule():
                schedule_callback(segment_callback(pipe_reader, pipe_name), pipe_writer, pipe_name)
            loop.call_soon_threadsafe(do_schedule)

        # hold tasks since `loop.create_task` is a weak reference that gets GC'd
        live_tasks = set()
        live_pipes = set()
        live_tasks_lock = threading.Lock()

        def schedule_callback(coro, pipe_writer, pipe_name):
            task = loop.create_task(coro)
            with live_tasks_lock:
                live_tasks.add(task)
                live_pipes.add(pipe_writer)
                
            def task_done2(t: asyncio.Task, p):
                try:
                    t.result()
                except Exception as e:
                    logger.error(f"Task {pipe_name} crashed: {e}")
                with live_tasks_lock:
                    live_tasks.remove(t)
                    live_pipes.remove(p)
                    
            def task_done(t2: asyncio.Task):
                return task_done2(task, pipe_writer)
            task.add_done_callback(task_done)

        # Start the encoding process
        encode_thread = threading.Thread(
            target=encode_frames,
            args=(live_pipes, live_tasks_lock, frame_generator, sync_callback, get_metadata),
            kwargs={"audio_codec": "libopus"}
        )
        encode_thread.start()
        logger.debug("run_publish: encoder thread started")

        # Wait for encode thread to complete
        def joins():
            encode_thread.join()
        await asyncio.to_thread(joins)

        # wait for IO tasks to complete
        while True:
            with live_tasks_lock:
                current_tasks = list(live_tasks)
            if not current_tasks:
                break  # nothing left to wait on
            await asyncio.wait(current_tasks, return_when=asyncio.ALL_COMPLETED)

        logger.info("run_publish complete")

    except Exception as e:
        logger.error(f"run_publish got error {e}", exc_info=True)
        raise e
    finally:
        if publisher:
            await publisher.close()

def encode_frames(task_pipes, task_lock, frame_generator, sync_callback, get_metadata, **kwargs):
    """
    Encode frames for streaming. This is a simplified version that creates
    pipe connections for streaming data.
    """
    retry_count = 0
    last_retry_time = time.time()
    
    while retry_count < MAX_ENCODER_RETRIES:
        try:
            # Create a pipe for the encoded data
            read_fd, write_fd = os.pipe()
            pipe_reader = os.fdopen(read_fd, 'rb')
            pipe_writer = os.fdopen(write_fd, 'wb')
            
            # Schedule the callback to handle the pipe
            sync_callback(pipe_reader, pipe_writer, "encoder_pipe")
            
            # Simulate encoding by writing frame data to the pipe
            # In a real implementation, this would use FFmpeg or similar
            for frame_data in frame_generator():
                if frame_data is None:
                    break
                pipe_writer.write(frame_data)
                pipe_writer.flush()
                
            pipe_writer.close()
            break  # clean exit
            
        except Exception as exc:
            current_time = time.time()
            # Reset retry counter if enough time has elapsed
            if current_time - last_retry_time > ENCODER_RETRY_RESET_SECONDS:
                logger.info("Resetting encoder retry count")
                retry_count = 0
            retry_count += 1
            last_retry_time = current_time
            
            if retry_count < MAX_ENCODER_RETRIES:
                logger.exception(f"Error in encode_frames, retrying {retry_count}/{MAX_ENCODER_RETRIES}")
            else:
                logger.exception("Error in encode_frames, maximum retries reached")
                
            # close leftover writer ends of any pipes to prevent hanging
            pipe_count = 0
            total_pipes = 0
            with task_lock:
                pipes = list(task_pipes)
                total_pipes = len(pipes)
                for p in pipes:
                    try:
                        p.close()
                        pipe_count += 1
                    except Exception as e:
                        logger.exception("Error closing pipe on task list")
            logger.info(f"Closed pipes - {pipe_count}/{total_pipes}")

async def simple_frame_publisher(publish_url: str, frame_queue: asyncio.Queue):
    """
    Simplified frame publisher that takes frames from a queue and publishes them.
    
    Args:
        publish_url: URL to publish the stream to
        frame_queue: Queue containing frame data to publish
    """
    try:
        publisher = TricklePublisher(url=publish_url, mime_type="video/mp2t")  # MPEG-TS format
        
        logger.info(f"Starting frame publisher for {publish_url}")
        
        while True:
            try:
                frame_data = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                if frame_data is None:  # End of stream signal
                    break
                    
                async with await publisher.next() as segment:
                    await segment.write(frame_data)
                    
            except asyncio.TimeoutError:
                continue  # Keep trying to get frames
            except Exception as e:
                logger.error(f"Error publishing frame: {e}")
                break
                
        logger.info("Frame publisher finished")
        
    except Exception as e:
        logger.error(f"Frame publisher error: {e}")
    finally:
        if publisher:
            await publisher.close()

async def enhanced_segment_publisher(publish_url: str, segment_queue: asyncio.Queue, 
                                   add_metadata_headers: bool = True):
    """
    Enhanced segment publisher that publishes encoded segments with proper metadata and optimized timing.
    
    Args:
        publish_url: URL to publish the stream to
        segment_queue: Queue containing encoded segment data to publish (may be tuples or bytes)
        add_metadata_headers: Whether to extract and add metadata headers
    """
    try:
        publisher = TricklePublisher(url=publish_url, mime_type="video/mp2t")  # MPEG-TS format
        
        logger.info(f"Starting enhanced segment publisher for {publish_url}")
        segment_count = 0
        
        while True:
            try:
                # Reduced timeout for faster processing - don't wait too long for segments
                segment_item = await asyncio.wait_for(segment_queue.get(), timeout=0.1)
                if segment_item is None:  # End of stream signal
                    break
                
                # Handle both tuple format (segment_id, segment_data) and bytes format
                if isinstance(segment_item, tuple) and len(segment_item) == 2:
                    segment_id, segment_data = segment_item
                    use_custom_id = True
                    logger.debug(f"Received segment {segment_id} with ID preservation")
                else:
                    segment_data = segment_item
                    segment_id = segment_count
                    use_custom_id = False
                
                # Ensure segment_data is bytes
                if not isinstance(segment_data, bytes):
                    logger.error(f"Expected bytes for segment data, got {type(segment_data)}")
                    continue
                
                segment_count += 1
                
                # Extract metadata for headers if requested (only on bytes!)
                if add_metadata_headers and segment_data:
                    try:
                        metadata = TrickleMetadataExtractor.extract_segment_metadata(segment_data)
                        headers = TrickleMetadataExtractor.create_segment_headers(metadata, segment_id)
                        logger.debug(f"Segment {segment_id} metadata: {metadata}")
                        # Note: Headers would be set on the publisher if the TricklePublisher supported them
                        # For now, we log them for debugging
                    except Exception as e:
                        logger.warning(f"Failed to extract metadata for segment {segment_id}: {e}")
                
                # Publish the segment (ensure only bytes are passed!)
                if use_custom_id and hasattr(publisher, 'publish_segment_at_index'):
                    # Use specific segment ID publishing
                    await publisher.publish_segment_at_index(segment_data, segment_id)
                    logger.debug(f"Published segment {segment_id} with custom ID")
                else:
                    # Use regular sequential publishing
                    async with await publisher.next() as segment:
                        await segment.write(segment_data)  # Only pass bytes here!
                    
                    if use_custom_id:
                        logger.debug(f"Published segment {segment_id} sequentially")
                    else:
                        logger.debug(f"Published MPEG-TS segment {segment_count} ({len(segment_data)} bytes)")
                    
            except asyncio.TimeoutError:
                # Don't log timeout messages as they're normal when no segments are available
                continue  
            except Exception as e:
                logger.error(f"Error publishing segment: {e}")
                # Don't break the loop, continue trying to publish other segments
                continue
                
        logger.info(f"Enhanced segment publisher finished: {segment_count} segments published")
        
    except Exception as e:
        logger.error(f"Enhanced segment publisher error: {e}")
    finally:
        if publisher:
            await publisher.close()

async def high_throughput_segment_publisher(publish_url: str, segment_queue: asyncio.Queue, 
                                          max_fps: float = 30.0, skip_frame_on_backlog: bool = True):
    """
    High-throughput segment publisher optimized for maximum FPS and minimal latency.
    
    Implements frame skipping when queue backs up to maintain real-time performance.
    
    Args:
        publish_url: URL to publish the stream to
        segment_queue: Queue containing encoded segment data to publish
        max_fps: Maximum frames per second to publish
        skip_frame_on_backlog: Whether to skip segments when queue backs up
    """
    try:
        publisher = TricklePublisher(url=publish_url, mime_type="video/mp2t")
        
        logger.info(f"Starting high-throughput segment publisher for {publish_url} (max {max_fps} fps)")
        
        segment_count = 0
        skipped_count = 0
        min_segment_interval = 1.0 / max_fps  # Minimum time between segments
        last_publish_time = 0
        
        while True:
            try:
                # Very short timeout to maximize responsiveness
                segment_item = await asyncio.wait_for(segment_queue.get(), timeout=0.05)
                if segment_item is None:  # End of stream signal
                    break
                
                current_time = time.time()
                
                # Handle both tuple and bytes format
                if isinstance(segment_item, tuple) and len(segment_item) == 2:
                    segment_id, segment_data = segment_item
                else:
                    segment_data = segment_item
                    segment_id = segment_count
                
                # Ensure segment_data is bytes
                if not isinstance(segment_data, bytes):
                    logger.error(f"Expected bytes for segment data, got {type(segment_data)}")
                    continue
                
                # Check if we should skip this segment due to rate limiting or backlog
                time_since_last = current_time - last_publish_time
                queue_size = segment_queue.qsize()
                
                if skip_frame_on_backlog and queue_size > 3:
                    # Queue is backing up, skip this segment to catch up
                    skipped_count += 1
                    logger.debug(f"Skipping segment {segment_id} due to queue backlog ({queue_size} segments)")
                    continue
                
                if time_since_last < min_segment_interval and last_publish_time > 0:
                    # Publishing too fast, skip this segment to maintain frame rate
                    skipped_count += 1
                    logger.debug(f"Skipping segment {segment_id} to maintain {max_fps} fps limit")
                    continue
                
                # Publish the segment
                try:
                    async with await publisher.next() as segment:
                        await segment.write(segment_data)
                    
                    segment_count += 1
                    last_publish_time = current_time
                    
                    # Log progress periodically
                    if segment_count % 30 == 0:  # Every 30 segments
                        actual_fps = 1.0 / time_since_last if time_since_last > 0 else 0
                        logger.info(f"Published {segment_count} segments (skipped {skipped_count}), "
                                  f"current FPS: {actual_fps:.1f}, queue size: {queue_size}")
                    
                except Exception as e:
                    logger.error(f"Error publishing segment {segment_id}: {e}")
                    continue
                    
            except asyncio.TimeoutError:
                # Normal when no segments available, continue
                continue
            except Exception as e:
                logger.error(f"Error in high-throughput publisher loop: {e}")
                continue
                
        logger.info(f"High-throughput segment publisher finished: {segment_count} segments published, {skipped_count} skipped")
        
    except Exception as e:
        logger.error(f"High-throughput segment publisher error: {e}")
    finally:
        if publisher:
            await publisher.close() 
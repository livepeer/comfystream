import time
import asyncio
import logging
import os
import threading
from typing import Callable, Any, Optional

from .trickle_publisher import TricklePublisher

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
        publisher = TricklePublisher(url=publish_url, mime_type="video/mp2t")
        
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
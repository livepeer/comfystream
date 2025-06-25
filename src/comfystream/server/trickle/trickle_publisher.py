import asyncio
import aiohttp
import logging
from contextlib import asynccontextmanager
import time
from typing import Optional, Dict, Any, AsyncIterator

logger = logging.getLogger(__name__)

class TricklePublisher:
    def __init__(self, url: str, mime_type: str, start_idx: int = 0):
        self.url = url
        self.mime_type = mime_type
        self.idx = start_idx  # Allow setting starting index to match input stream
        self.next_writer = None
        self.lock = asyncio.Lock()  # Lock to manage concurrent access
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()

    def streamIdx(self):
        return f"{self.url}/{self.idx}"

    async def preconnect(self):
        """Preconnect to the server by initiating a POST request to the current index."""
        url = self.streamIdx()
        logger.info(f"Preconnecting to URL: {url}")
        try:
            # we will be incrementally writing data into this queue
            queue = asyncio.Queue()
            asyncio.create_task(self._run_post(url, queue))
            return queue
        except aiohttp.ClientError as e:
            logger.error(f"Failed to complete POST for {url}: {e}")
            return None

    async def _run_post(self, url, queue):
        try:
            if not self.session:
                logger.error("Session is None, cannot perform POST")
                return None
            resp = await self.session.post(
                url,
                headers={'Connection': 'close', 'Content-Type': self.mime_type},
                data=self._stream_data(queue)
            )
            # TODO propagate errors?
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Trickle POST failed {self.streamIdx()}, status code: {resp.status}, msg: {body}")
        except Exception as e:
            logger.error(f"Trickle POST  exception {self.streamIdx()} - {e}")
        return None

    async def _run_delete(self):
        try:
            if self.session:
                await self.session.delete(self.url)
        except Exception:
            logger.error(f"Error sending trickle delete request", exc_info=True)

    async def _stream_data(self, queue):
        """Stream data from the queue for the POST request."""
        while True:
            chunk = await queue.get()
            if chunk is None:  # Stop signal
                break
            yield chunk

    async def next(self):
        """Start or retrieve a pending POST request and preconnect for the next segment."""
        async with self.lock:
            if self.next_writer is None:
                logger.info(f"No pending connection, preconnecting {self.streamIdx()}...")
                self.next_writer = await self.preconnect()

            writer = self.next_writer
            self.next_writer = None

            # Set up the next POST in the background
            asyncio.create_task(self._preconnect_next_segment())

        if writer is None:
            logger.error("No writer available for segment")
            # Create a dummy queue to avoid None issues
            writer = asyncio.Queue()
        
        return SegmentWriter(writer)

    async def _preconnect_next_segment(self):
        """Preconnect to the next POST in the background."""
        logger.info(f"Setting up next connection for {self.streamIdx()}")
        async with self.lock:
            if self.next_writer is not None:
                return
            self.idx += 1  # Increment the index for the next POST
            next_writer = await self.preconnect()
            if next_writer:
                self.next_writer = next_writer

    async def close(self):
        """Close the session when done."""
        logger.info(f"Closing {self.url}")
        async with self.lock:
            if self.next_writer:
                s = SegmentWriter(self.next_writer)
                await s.close()
                self.next_writer = None
            if self.session:
                try:
                    await self._run_delete()
                    await self.session.close()
                except Exception:
                    logger.error(f"Error closing trickle publisher", exc_info=True)
                finally:
                    self.session = None

    def set_segment_index(self, idx: int):
        """Set the current segment index to match input stream numbering"""
        self.idx = idx
        logger.info(f"Set publisher segment index to {idx}")

    async def publish_segment_at_index(self, segment_data: bytes, segment_id: int):
        """Publish a segment at a specific index"""
        # Validate input to prevent tuple errors in HTTP stream
        if not isinstance(segment_data, bytes):
            raise TypeError(f"segment_data must be bytes, got {type(segment_data)}")
        
        # Temporarily set the index
        old_idx = self.idx
        self.idx = segment_id
        
        try:
            async with await self.next() as segment:
                # CRITICAL: Only pass bytes to avoid HTTP stream tuple errors
                await segment.write(segment_data)
            logger.debug(f"Published segment at index {segment_id}")
        finally:
            # Restore index (though it will be incremented by next() call)
            pass

class SegmentWriter:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def write(self, data):
        """Write data to the current segment."""
        # CRITICAL: Validate that only bytes reach the HTTP stream
        if not isinstance(data, bytes):
            raise TypeError(f"SegmentWriter.write() requires bytes, got {type(data)}. "
                          f"This prevents HTTP stream tuple errors.")
        
        if self.queue:
            await self.queue.put(data)

    async def close(self):
        """Ensure the request is properly closed when done."""
        if self.queue:
            await self.queue.put(None)  # Send None to signal end of data

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the connection."""
        await self.close()

async def enhanced_segment_publisher(
    base_url: str, 
    segment_queue: asyncio.Queue, 
    add_metadata_headers: bool = False,
    target_fps: float = 24.0,  # Expected playback FPS - match segmenter
    segment_duration: float = 1.0  # Duration per segment in seconds
) -> None:
    """
    Enhanced segment publisher with proper timing control and segment ID preservation.
    
    Publishes segments at the correct rate and preserves segment numbering
    to maintain correspondence with input stream.
    """
    try:
        logger.info(f"Enhanced segment publisher starting for {base_url}")
        logger.info(f"Target timing: {target_fps}fps, {segment_duration}s per segment")
        
        segment_interval = segment_duration  # Time between segment publications
        last_publish_time = None
        published_count = 0
        
        # Don't set a start index - will be determined by first segment received
        publisher = TricklePublisher(url=base_url, mime_type="video/mp2t")
        first_segment = True
        
        while True:
            try:
                # Get next segment data (should be tuple of (segment_id, segment_data) if available)
                segment_item = await asyncio.wait_for(
                    segment_queue.get(), 
                    timeout=5.0
                )
                
                if segment_item is None:
                    logger.info("End of stream signal received")
                    break
                
                # Handle both old format (just bytes) and new format (tuple with ID)
                if isinstance(segment_item, tuple) and len(segment_item) == 2:
                    segment_id, segment_data = segment_item
                    use_segment_id = True
                else:
                    segment_data = segment_item  # This should be bytes
                    segment_id = published_count
                    use_segment_id = False
                
                # Ensure segment_data is bytes
                if not isinstance(segment_data, bytes):
                    logger.error(f"Expected bytes for segment data, got {type(segment_data)}")
                    continue
                
                # Calculate timing for smooth delivery
                current_time = time.time()
                
                if last_publish_time is not None:
                    # Calculate time since last publish
                    time_since_last = current_time - last_publish_time
                    
                    # If we're publishing too fast, wait to maintain proper timing
                    if time_since_last < segment_interval:
                        wait_time = segment_interval - time_since_last
                        logger.debug(f"Pacing: waiting {wait_time:.3f}s for proper segment timing")
                        await asyncio.sleep(wait_time)
                        current_time = time.time()
                
                # Publish segment with proper timing control
                try:
                    if use_segment_id and hasattr(publisher, 'publish_segment_at_index'):
                        # Use specific segment ID to maintain correspondence
                        # CRITICAL: Only pass bytes to avoid HTTP stream errors
                        await publisher.publish_segment_at_index(segment_data, segment_id)
                        logger.debug(f"Published segment {segment_id} with ID preservation")
                    else:
                        # Fallback to sequential publishing
                        if first_segment and use_segment_id:
                            # Set starting index to match first segment
                            publisher.set_segment_index(segment_id)
                            first_segment = False
                        
                        # CRITICAL: Only pass bytes to segment.write() to avoid tuple errors
                        async with await publisher.next() as segment:
                            await segment.write(segment_data)  # segment_data is guaranteed to be bytes
                        
                        if use_segment_id:
                            logger.debug(f"Published segment {segment_id} sequentially")
                    
                    published_count += 1
                    last_publish_time = current_time
                    
                    # Log progress with timing info
                    if published_count % 10 == 0:
                        avg_interval = (current_time - (last_publish_time - segment_interval * 9)) / 10 if published_count >= 10 else segment_interval
                        logger.info(f"Published {published_count} segments with {avg_interval:.3f}s avg interval (target: {segment_interval:.3f}s)")
                    
                except Exception as e:
                    logger.error(f"Error publishing segment {segment_id if use_segment_id else published_count}: {e}")
                    continue
                
            except asyncio.TimeoutError:
                logger.debug("No segments available, checking for end of stream...")
                continue
            except Exception as e:
                logger.error(f"Error in segment publisher loop: {e}")
                break
        
        logger.info(f"Enhanced segment publisher finished: {published_count} segments published")
        
    except Exception as e:
        logger.error(f"Enhanced segment publisher error: {e}")
    finally:
        try:
            await publisher.close()
        except:
            pass 
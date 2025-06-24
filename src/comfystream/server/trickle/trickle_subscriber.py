import asyncio
import aiohttp
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class TrickleSubscriber:
    """
    Trickle subscriber implementation following the http-trickle protocol.
    Subscribes to a trickle stream using GET requests to receive streaming segments.
    
    Implements trickle discovery protocol using sequence number -1 to find
    the most recent available segment before starting sequential subscription.
    """
    
    def __init__(self, url: str):
        self.url = url
        self.session = None
        self.segment_idx = None  # Will be discovered using -1 protocol
        self.discovered = False
        
    async def __aenter__(self):
        """Enter context manager."""
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False),
            timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()
        
    async def next(self):
        """Get the next segment from the trickle stream using GET request."""
        try:
            if self.session is None:
                logger.error("Session is None, cannot subscribe")
                return None
            
            # First time: discover the most recent segment using -1 protocol
            if not self.discovered:
                return await self._discover_latest_segment()
            
            # Ensure segment_idx is set after discovery
            if self.segment_idx is None:
                logger.error("Segment index not set - discovery may have failed")
                return None
                
            # Build segment URL following trickle protocol: baseurl/segment_index
            segment_url = f"{self.url}/{self.segment_idx}"
            logger.debug(f"Requesting trickle segment: {segment_url}")
            
            # Use GET request to subscribe to the segment (trickle protocol)
            response = await self.session.get(segment_url)
            
            if response.status == 200:
                logger.debug(f"Successfully subscribed to segment {self.segment_idx}")
                current_idx = self.segment_idx
                # Increment for next segment
                self.segment_idx += 1
                return SegmentReader(response, current_idx)
            elif response.status == 404:
                # No more segments available yet, wait and retry (normal for live streams)
                logger.debug(f"Segment {self.segment_idx} not available yet")
                response.close()
                return None
            elif response.status == 470:
                # Segment too old - need to catch up to current segments
                logger.warning(f"Segment {self.segment_idx} too old (470), re-discovering latest")
                response.close()
                self.discovered = False  # Re-discover latest segment
                
                # Reset segment index so discovery can set it properly
                self.segment_idx = None
                return await self._discover_latest_segment()
            else:
                logger.error(f"Failed to subscribe to {segment_url}, status: {response.status}")
                response.close()
                return None
                
        except asyncio.TimeoutError:
            logger.debug(f"Timeout waiting for segment {self.segment_idx}")
            return None
        except Exception as e:
            segment_desc = self.segment_idx if self.segment_idx is not None else "unknown"
            logger.error(f"Error subscribing to {self.url}/{segment_desc}: {e}")
            return None
    
    async def _discover_latest_segment(self):
        """
        Discover the most recent segment using trickle protocol sequence number -1.
        
        From the trickle protocol spec:
        "Subscribers can initiate a subscribe with a seq of -1 to retrieve the most recent publish"
        """
        try:
            if self.session is None:
                logger.error("Session is None, cannot discover latest segment")
                return None
                
            # Use -1 to get the most recent segment (trickle discovery protocol)
            discovery_url = f"{self.url}/-1"
            logger.info(f"Discovering latest trickle segment: {discovery_url}")
            
            response = await self.session.get(discovery_url)
            
            if response.status == 200:
                # Success! We got the most recent segment
                logger.info(f"Discovered latest segment via {discovery_url}")
                self.discovered = True
                
                # Extract the current segment number from Lp-Trickle-Seq header
                # This header contains the sequence number of the segment returned by /-1
                lp_trickle_seq = response.headers.get('Lp-Trickle-Seq')
                
                if lp_trickle_seq:
                    try:
                        current_idx = int(lp_trickle_seq)
                        self.segment_idx = current_idx + 1  # Next segment after current
                        logger.info(f"Discovery: Lp-Trickle-Seq={current_idx}, next segment will be {self.segment_idx}")
                        # Use the actual segment index from header
                        current_segment = SegmentReader(response, current_idx)
                    except ValueError:
                        logger.warning(f"Invalid Lp-Trickle-Seq header: {lp_trickle_seq}")
                        self.segment_idx = 1  # Start from 1 instead of 0
                        current_segment = SegmentReader(response, -1)
                else:
                    # Simple fallback: Since /-1 gave us the LATEST segment, 
                    # the next segment to request should be recent (start from 1)
                    # Let 470 error handling guide us to the right segment naturally
                    self.segment_idx = 1
                    logger.info(f"Discovery: No Lp-Trickle-Seq header provided, starting from segment {self.segment_idx} (let 470 handling guide us)")
                    # Use -1 as placeholder for discovery segment
                    current_segment = SegmentReader(response, -1)
                
                return current_segment
                
            elif response.status == 404:
                # No segments available at all
                logger.warning("No segments available for discovery (404)")
                response.close()
                return None
                
            else:
                logger.error(f"Failed to discover latest segment, status: {response.status}")
                response.close()
                return None
                
        except Exception as e:
            logger.error(f"Error discovering latest segment: {e}")
            return None
    

        
    async def close(self):
        """Close the session when done."""
        if self.session:
            await self.session.close()
            self.session = None

class SegmentReader:
    def __init__(self, response, segment_idx: int = 0):
        self.response = response
        self.segment_idx = segment_idx
        self._closed = False
        self._total_read = 0
        
    async def read(self, size: int = 8192):
        """Read data from the trickle segment."""
        if self._closed:
            return b""
            
        try:
            data = await self.response.content.read(size)
            
            if data:
                self._total_read += len(data)
                logger.debug(f"Read {len(data)} bytes from segment {self.segment_idx} (total: {self._total_read})")
                return data
            else:
                # End of this segment
                logger.debug(f"Segment {self.segment_idx} completed, read {self._total_read} total bytes")
                self._closed = True
                return b""
                
        except Exception as e:
            logger.error(f"Error reading from segment {self.segment_idx}: {e}")
            self._closed = True
            return b""
            
    async def close(self):
        """Close the segment reader."""
        if self.response and not self._closed:
            logger.debug(f"Closing segment {self.segment_idx}")
            self.response.close()
            self._closed = True 
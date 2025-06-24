import asyncio
import aiohttp
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class TrickleSubscriber:
    """
    Simple trickle subscriber implementation for ComfyStream.
    Primarily used for receiving streaming data.
    """
    
    def __init__(self, url: str):
        self.url = url
        self.session = None
        self.response = None
        
    async def __aenter__(self):
        """Enter context manager."""
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
        return self
        
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()
        
    async def next(self):
        """Get the next segment from the stream."""
        if not self.response:
            try:
                self.response = await self.session.get(self.url)
                if self.response.status != 200:
                    logger.error(f"Failed to subscribe to {self.url}, status: {self.response.status}")
                    return None
            except Exception as e:
                logger.error(f"Error subscribing to {self.url}: {e}")
                return None
                
        return SegmentReader(self.response)
        
    async def close(self):
        """Close the session when done."""
        if self.response:
            self.response.close()
            self.response = None
        if self.session:
            await self.session.close()
            self.session = None

class SegmentReader:
    def __init__(self, response):
        self.response = response
        
    async def read(self, size: int = 8192):
        """Read data from the segment."""
        try:
            return await self.response.content.read(size)
        except Exception as e:
            logger.error(f"Error reading segment data: {e}")
            return None
            
    async def close(self):
        """Close the segment reader."""
        if self.response:
            self.response.close() 
"""General utility functions."""

import asyncio
import random
import types
import logging
from aiohttp import web
from typing import List, Tuple
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# Original issue: https://github.com/aiortc/aioice/pull/63
# Copied from: https://github.com/toverainc/willow-inference-server/pull/17/files
def patch_loop_datagram(local_ports: List[int]):
    loop = asyncio.get_event_loop()
    if getattr(loop, "_patch_done", False):
        return

    # Monkey patch aiortc to control ephemeral ports
    old_create_datagram_endpoint = loop.create_datagram_endpoint

    async def create_datagram_endpoint(
        self, protocol_factory, local_addr: Tuple[str, int] = None, **kwargs
    ):
        # if port is specified just use it
        if local_addr and local_addr[1]:
            return await old_create_datagram_endpoint(
                protocol_factory, local_addr=local_addr, **kwargs
            )
        if local_addr is None:
            return await old_create_datagram_endpoint(
                protocol_factory, local_addr=None, **kwargs
            )
        # if port is not specified make it use our range
        ports = list(local_ports)
        random.shuffle(ports)
        for port in ports:
            try:
                ret = await old_create_datagram_endpoint(
                    protocol_factory, local_addr=(local_addr[0], port), **kwargs
                )
                logger.debug(f"create_datagram_endpoint chose port {port}")
                return ret
            except OSError as exc:
                if port == ports[-1]:
                    # this was the last port, give up
                    raise exc
        raise ValueError("local_ports must not be empty")

    loop.create_datagram_endpoint = types.MethodType(create_datagram_endpoint, loop)
    loop._patch_done = True


def add_prefix_to_app_routes(app: web.Application, prefix: str):
    """Add a prefix to all routes in the given application.

    Args:
        app: The web application whose routes will be prefixed.
        prefix: The prefix to add to all routes.
    """
    prefix = prefix.rstrip("/")
    for route in list(app.router.routes()):
        new_path = prefix + route.resource.canonical
        app.router.add_route(route.method, new_path, route.handler)


@asynccontextmanager
async def temporary_log_level(logger_name: str, level: int):
    """Temporarily set the log level of a logger.

    Args:
        logger_name: The name of the logger to set the level for.
        level: The log level to set.
    """
    if level is not None:
        logger = logging.getLogger(logger_name)
        original_level = logger.level
        logger.setLevel(level)
    try:
        yield
    finally:
        if level is not None:
            logger.setLevel(original_level)


class ComfyStreamTimeoutFilter(logging.Filter):
    """
    Custom logging filter to suppress verbose ComfyUI execution stack traces 
    for ComfyStream timeout exceptions.
    
    This prevents the extensive stack traces that ComfyUI logs when our LoadTensor
    and LoadAudioTensor nodes timeout waiting for input frames, which is expected
    behavior during stream switching and warmup scenarios.
    """
    
    def filter(self, record):
        """
        Filter out ComfyUI execution error logs for ComfyStream timeout exceptions.
        
        Args:
            record: The log record to potentially filter
            
        Returns:
            False to suppress the log, True to allow it
        """
        # Only filter ERROR level messages from ComfyUI execution system
        if record.levelno != logging.ERROR:
            return True
            
        # Check if this is from ComfyUI execution system
        if not (record.name.startswith("comfy") and ("execution" in record.name or record.name == "comfy")):
            return True
            
        # Get the full message including any exception info
        message = record.getMessage()
        
        # Check if this is a ComfyStream timeout-related error
        timeout_indicators = [
            "ComfyStreamInputTimeoutError",
            "ComfyStreamAudioBufferError", 
            "No video frames available",
            "No audio frames available", 
            "Audio stream interrupted",
            "insufficient data available",
            "ComfyStream may not be receiving input"
        ]
        
        # Suppress if any timeout indicator is found in the message
        for indicator in timeout_indicators:
            if indicator in message:
                return False
                
        # Also check the exception info if present
        if record.exc_info and record.exc_info[1]:
            exc_str = str(record.exc_info[1])
            for indicator in timeout_indicators:
                if indicator in exc_str:
                    return False
                    
        # Check if the record has exc_text (formatted exception)
        if hasattr(record, 'exc_text') and record.exc_text:
            for indicator in timeout_indicators:
                if indicator in record.exc_text:
                    return False
                    
        return True

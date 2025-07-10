"""General utility functions."""

import asyncio
import random
import types
import json
import logging
from aiohttp import web
from typing import List, Tuple, Optional, Any, Union, Dict, List
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


def parse_prompts_parameter(prompts_param: Any) -> Union[Dict[Any, Any], List[Dict[Any, Any]]]:
    """
    Parse prompts parameter, handling both dict/list and JSON string formats.
    
    Args:
        prompts_param: The prompts parameter which can be:
            - A dictionary (workflow object)
            - A list of dictionaries
            - A JSON string representing a workflow or list of workflows
    
    Returns:
        Parsed prompts as dict or list of dicts
    
    Raises:
        ValueError: If the prompts parameter cannot be parsed
    """
    if isinstance(prompts_param, (dict, list)):
        # Already parsed, return as-is
        return prompts_param
    elif isinstance(prompts_param, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(prompts_param)
            if isinstance(parsed, (dict, list)):
                return parsed
            else:
                raise ValueError(f"Parsed JSON is not a dict or list, got {type(parsed)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts parameter: {e}")
    else:
        raise ValueError(f"Prompts parameter must be dict, list, or JSON string, got {type(prompts_param)}")


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


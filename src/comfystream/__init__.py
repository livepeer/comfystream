# Import comfy_loader first to setup namespace
from . import comfy_loader

# Only import server components if not in vanilla custom node context
if comfy_loader.should_load_comfystream_components():
    try:
        from .client import ComfyStreamClient
        from .pipeline import Pipeline
        from .server.utils import temporary_log_level
        from .server.utils import FPSMeter
        from .server.metrics import MetricsManager, StreamStatsManager

        __all__ = [
            'ComfyStreamClient',
            'Pipeline',
            'temporary_log_level',
            'FPSMeter',
            'MetricsManager',
            'StreamStatsManager'
        ]
    except ImportError as e:
        # If server components can't be imported, just provide empty exports
        __all__ = []
else:
    __all__ = []

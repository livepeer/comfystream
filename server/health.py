"""
ComfyStream-specific health management.

This module provides ComfyStream-specific health state management that extends
the base StreamHealthManager from pytrickle with WebRTC and Trickle stream tracking.
"""

from pytrickle.health import StreamHealthManager
from typing import Dict, Any

class ComfyStreamHealthManager(StreamHealthManager):
    """ComfyStream-specific health manager that tracks WebRTC and Trickle streams separately."""
    
    def __init__(self):
        super().__init__("comfystream-service")
        self.active_webrtc_streams = 0
        self.active_trickle_streams = 0
        
    def update_webrtc_streams(self, count: int):
        """Update count of active WebRTC streams."""
        self.active_webrtc_streams = count
        self.active_streams = self.active_webrtc_streams + self.active_trickle_streams
        self._update_state()
        
    def update_trickle_streams(self, count: int):
        """Update count of active trickle streams."""
        self.active_trickle_streams = count
        self.active_streams = self.active_webrtc_streams + self.active_trickle_streams
        self._update_state()
        
    def get_status(self) -> Dict[str, Any]:
        """Get current health status with ComfyStream-specific fields."""
        status = super().get_status()
        status.update({
            "active_webrtc_streams": self.active_webrtc_streams,
            "active_trickle_streams": self.active_trickle_streams,
        })
        return status

# Alias for backward compatibility
HealthStateManager = ComfyStreamHealthManager
"""
ComfyStream-specific health management.

This module provides ComfyStream-specific health state management that extends
the base HealthManager from pytrickle with WebRTC and Trickle stream tracking.
"""

from typing import Dict, Any
from pytrickle.health import StreamHealthManager


class ComfyStreamHealthManager(StreamHealthManager):
    """ComfyStream-specific health manager that tracks WebRTC and Trickle streams separately."""
    
    def __init__(self):
        super().__init__("comfystream-service")
        self.active_webrtc_streams = 0
        self.active_trickle_streams = 0
        
    def update_webrtc_streams(self, count: int):
        """Update count of active WebRTC streams."""
        self.active_webrtc_streams = count
        total_streams = self.active_webrtc_streams + self.active_trickle_streams
        self.update_active_streams(total_streams)
        
    def update_trickle_streams(self, count: int):
        """Update count of active trickle streams."""
        self.active_trickle_streams = count
        total_streams = self.active_webrtc_streams + self.active_trickle_streams
        self.update_active_streams(total_streams)
        
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current health state with ComfyStream-specific fields."""
        state = super().get_pipeline_state()
        state.update({
            "active_webrtc_streams": self.active_webrtc_streams,
            "active_trickle_streams": self.active_trickle_streams,
            "total_streams": self.active_webrtc_streams + self.active_trickle_streams,
        })
        return state

# Alias for backward compatibility
HealthStateManager = ComfyStreamHealthManager
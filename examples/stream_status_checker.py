#!/usr/bin/env python3
"""
Stream Status Checker for ComfyStream

This script provides easy ways to check the status of WHIP/WHEP streams
and processing readiness.
"""

import requests
import json
import time
import sys
from datetime import datetime

class StreamStatusChecker:
    """Check and monitor ComfyStream status."""
    
    def __init__(self, base_url="http://localhost:8889"):
        self.base_url = base_url.rstrip('/')
    
    def get_processing_status(self):
        """Get comprehensive processing status."""
        try:
            response = requests.get(f"{self.base_url}/processing/status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_whip_status(self):
        """Get WHIP (ingestion) status."""
        try:
            response = requests.get(f"{self.base_url}/whip-stats")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_whep_status(self):
        """Get WHEP (subscription) status."""
        try:
            response = requests.get(f"{self.base_url}/whep-stats")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_stream_stats(self):
        """Get general stream statistics."""
        try:
            response = requests.get(f"{self.base_url}/streams/stats")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def print_status_summary(self):
        """Print a formatted status summary."""
        print("=" * 60)
        print(f"ComfyStream Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Processing status
        status = self.get_processing_status()
        if "error" in status:
            print(f"❌ Error getting status: {status['error']}")
            return
        
        # Main status
        ready_icon = "✅" if status.get("processing_ready") else "⏳"
        print(f"{ready_icon} Processing Ready: {status.get('processing_ready', False)}")
        print(f"📝 Message: {status.get('message', 'No message')}")
        print()
        
        # Session counts
        print("📊 Session Summary:")
        print(f"  📥 WHIP Sessions (Incoming): {status.get('whip_sessions', 0)}")
        print(f"  📤 WHEP Sessions (Outgoing): {status.get('whep_sessions', 0)}")
        print(f"  🔄 Active Pipelines: {status.get('active_pipelines', 0)}")
        print(f"  🎬 Frames Available: {status.get('frames_available', False)}")
        print()
        
        # Detailed session info
        details = status.get("details", {})
        
        # WHIP sessions detail
        whip_sessions = details.get("whip_sessions", {})
        if whip_sessions:
            print("📥 WHIP Sessions (Incoming Streams):")
            for session_id, session_info in whip_sessions.items():
                state = session_info.get("connection_state", "unknown")
                has_video = "📹" if session_info.get("has_video") else ""
                has_audio = "🔊" if session_info.get("has_audio") else ""
                created = session_info.get("created_at", 0)
                duration = time.time() - created if created else 0
                print(f"  • {session_id[:8]}... ({state}) {has_video}{has_audio} [{duration:.0f}s]")
        else:
            print("📥 No active WHIP sessions")
        print()
        
        # WHEP sessions detail
        whep_sessions = details.get("whep_sessions", {})
        if whep_sessions:
            print("📤 WHEP Sessions (Outgoing Subscribers):")
            for session_id, session_info in whep_sessions.items():
                state = session_info.get("connection_state", "unknown")
                has_video = "📹" if session_info.get("has_video") else ""
                has_audio = "🔊" if session_info.get("has_audio") else ""
                created = session_info.get("created_at", 0)
                duration = time.time() - created if created else 0
                print(f"  • {session_id[:8]}... ({state}) {has_video}{has_audio} [{duration:.0f}s]")
        else:
            print("📤 No active WHEP sessions")
        print()
    
    def wait_for_processing_ready(self, timeout=60, check_interval=2):
        """Wait for processing to be ready."""
        print(f"⏳ Waiting for processing to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_processing_status()
            
            if "error" in status:
                print(f"❌ Error: {status['error']}")
                time.sleep(check_interval)
                continue
            
            if status.get("processing_ready"):
                print(f"✅ Processing is ready! {status.get('message')}")
                return True
            
            print(f"⏳ {status.get('message')} (checking again in {check_interval}s)")
            time.sleep(check_interval)
        
        print(f"⏱️ Timeout waiting for processing to be ready")
        return False
    
    def monitor_status(self, interval=5):
        """Continuously monitor status."""
        print(f"👁️ Monitoring stream status every {interval} seconds (Ctrl+C to stop)")
        print()
        
        try:
            while True:
                self.print_status_summary()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")


def main():
    """Main function with CLI interface."""
    if len(sys.argv) < 2:
        print("Usage: python stream_status_checker.py <command> [options]")
        print()
        print("Commands:")
        print("  status              - Show current status once")
        print("  monitor [interval]  - Monitor status continuously")
        print("  wait [timeout]      - Wait for processing to be ready")
        print("  whip               - Show only WHIP status")
        print("  whep               - Show only WHEP status")
        print()
        print("Examples:")
        print("  python stream_status_checker.py status")
        print("  python stream_status_checker.py monitor 10")
        print("  python stream_status_checker.py wait 30")
        return
    
    command = sys.argv[1].lower()
    checker = StreamStatusChecker()
    
    if command == "status":
        checker.print_status_summary()
    
    elif command == "monitor":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        checker.monitor_status(interval)
    
    elif command == "wait":
        timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        ready = checker.wait_for_processing_ready(timeout)
        sys.exit(0 if ready else 1)
    
    elif command == "whip":
        whip_status = checker.get_whip_status()
        print("WHIP Status:")
        print(json.dumps(whip_status, indent=2))
    
    elif command == "whep":
        whep_status = checker.get_whep_status()
        print("WHEP Status:")
        print(json.dumps(whep_status, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
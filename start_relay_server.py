#!/usr/bin/env python3
"""
Start ComfyStream Relay Server

This script starts the ComfyStream relay server that's needed for trickle streaming.
"""

import subprocess
import sys
import os
import time
import argparse
import signal

def start_comfystream_relay(
    host="0.0.0.0",
    port=9876,
    media_ports="5678,5679,5680,5681",
    workspace="/workspace/ComfyUI"
):
    """Start the ComfyStream relay server."""
    
    python_exec = "/workspace/miniconda3/envs/comfystream/bin/python"
    server_script = "/workspace/comfystream/server/main.py"
    
    if not os.path.exists(server_script):
        print(f"❌ Server script not found: {server_script}")
        return None
    
    cmd = [
        python_exec,
        server_script,
        f"--host={host}",
        f"--port={port}",
        f"--media-ports={media_ports}",
        f"--workspace={workspace}"
    ]
    
    print(f"🚀 Starting ComfyStream relay server...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Host: {host}:{port}")
    print(f"   Media ports: {media_ports}")
    print(f"   Workspace: {workspace}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"✅ Server started with PID: {process.pid}")
        print("📡 Server output:")
        print("-" * 40)
        
        # Monitor server output
        while True:
            try:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    
                    # Check for server ready indicators
                    if "listening on" in output.lower() or "server running" in output.lower():
                        print("✅ Server appears to be ready!")
                        
            except KeyboardInterrupt:
                print("\n⛔ Stopping server...")
                process.terminate()
                process.wait()
                break
                
        return process
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Start ComfyStream relay server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9876, help="Server port")
    parser.add_argument("--media-ports", default="5678,5679,5680,5681", help="Media ports")
    parser.add_argument("--workspace", default="/workspace/ComfyUI", help="ComfyUI workspace")
    
    args = parser.parse_args()
    
    print("🎛️  ComfyStream Relay Server Starter")
    print("=" * 40)
    
    # Check if ComfyUI workspace exists
    if not os.path.exists(args.workspace):
        print(f"⚠️  ComfyUI workspace not found: {args.workspace}")
        print("   You may need to clone ComfyUI or adjust the workspace path")
    
    # Start the server
    process = start_comfystream_relay(
        host=args.host,
        port=args.port,
        media_ports=args.media_ports,
        workspace=args.workspace
    )
    
    if process:
        print(f"\n🎯 Server started successfully!")
        print(f"   Health check: http://{args.host}:{args.port}/health")
        print(f"   Media ports: {args.media_ports}")
        print("\n💡 You can now run the trickle integration example:")
        print("   python examples/comfystream_trickle_example.py")
        print("\n⛔ Press Ctrl+C to stop the server")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n⛔ Stopping server...")
            process.terminate()
            process.wait()
    else:
        print("❌ Failed to start server")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

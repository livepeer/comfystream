import csv
import os
import time
from typing import Optional, Dict, Any
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def log_frame_timing(
    frame_id: Optional[int],
    frame_received_time: Optional[float],
    frame_processed_time: Optional[float],
    client_index: Optional[int] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    csv_path: str = "frame_logs.csv"
):
    """
    Log frame timing information to a CSV file.
    Args:
        frame_id: The unique identifier for the frame.
        frame_received_time: Timestamp when the frame was received by pipeline.
        frame_processed_time: Timestamp when the frame was processed.
        client_index: Index of the client that processed this frame.
        additional_metadata: Any additional data to log (will be converted to string).
        csv_path: Path to the CSV file (default: 'frame_logs.csv').
    """
    latency = None
    if frame_received_time is not None and frame_processed_time is not None:
        latency = frame_processed_time - frame_received_time

    # Calculate absolute time
    current_time = time.time()

    # Convert additional metadata to string if present
    metadata_str = str(additional_metadata) if additional_metadata else None

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = [
                "log_timestamp", "frame_id", "frame_received_time", 
                "frame_processed_time", "latency_ms", "client_index", "metadata"
            ]
            writer.writerow(header)
        
        # Calculate latency in milliseconds for logging
        latency_ms = None
        if frame_received_time is not None and frame_processed_time is not None:
            latency_ms = (frame_processed_time - frame_received_time) * 1000

        writer.writerow([
            current_time, frame_id, frame_received_time, frame_processed_time, 
            latency_ms, client_index, metadata_str
        ])


def plot_frame_metrics(csv_path: str = "frame_logs.csv"):
    if not os.path.isfile(csv_path):
        print(f"CSV file '{csv_path}' not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"CSV file '{csv_path}' is empty.")
        return

    # Drop rows with missing times or frame_id
    df = df.dropna(subset=["frame_id", "frame_received_time", "frame_processed_time"])
    if df.empty:
        print("No valid timing data in CSV after dropping NA in essential time columns.")
        return

    # Sort by frame_id to ensure correct interval calculations
    df = df.sort_values("frame_id").reset_index(drop=True)
    
    # Calculate time since start of stream (first frame)
    stream_start_time = df["frame_received_time"].min()
    df["time_since_start"] = df["frame_received_time"] - stream_start_time
    
    # Calculate intervals between consecutive frames
    # Input interval: Time between a frame and the previous frame being received
    df["input_interval_s"] = df["frame_received_time"].diff()
    # Output interval: Time between a frame and the previous frame being processed
    df["output_interval_s"] = df["frame_processed_time"].diff()

    # Handle potential zero or negative intervals (e.g., from duplicate timestamps or sorting issues if not by frame_id)
    # These would lead to infinite or meaningless FPS, so set them to NaN.
    df.loc[df["input_interval_s"] <= 0, "input_interval_s"] = np.nan
    df.loc[df["output_interval_s"] <= 0, "output_interval_s"] = np.nan

    # Calculate FPS (Frames Per Second)
    # FPS is the reciprocal of the interval in seconds.
    df["input_fps"] = 1.0 / df["input_interval_s"]
    df["output_fps"] = 1.0 / df["output_interval_s"]
    
    # Rolling statistics for smoothing
    window = 30  # Increased window for smoother, more stable FPS and jitter
    df["input_fps_smooth"] = df["input_fps"].rolling(window, min_periods=1).mean()
    df["output_fps_smooth"] = df["output_fps"].rolling(window, min_periods=1).mean()
    
    # Jitter: Standard deviation of intervals (in milliseconds)
    # This measures the variation in frame arrival/processing times.
    df["input_jitter_ms"] = df["input_interval_s"].rolling(window, min_periods=1).std() * 1000
    df["output_jitter_ms"] = df["output_interval_s"].rolling(window, min_periods=1).std() * 1000

    # Latency: Time taken from frame reception to frame processing (in milliseconds)
    # Use pre-calculated 'latency_ms' if available and valid, otherwise recalculate.
    if "latency_ms" not in df.columns or df["latency_ms"].isnull().all():
        print("Recalculating latency_ms as it's not found or all null in CSV.")
        df["latency_ms"] = (df["frame_processed_time"] - df["frame_received_time"]) * 1000
    else:
        # Ensure it's numeric if it came from CSV
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors='coerce')

    df["latency_ms_smooth"] = df["latency_ms"].rolling(window, min_periods=1).mean()

    # Create visualization - reduced to 3 plots (removed frame distribution)
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # FPS plot
    axs[0].plot(df["time_since_start"], df["input_fps_smooth"], label="Input FPS (smooth)", color="blue", linewidth=2)
    axs[0].plot(df["time_since_start"], df["output_fps_smooth"], label="Output FPS (smooth)", color="green", linewidth=2)
    axs[0].set_ylabel("FPS")
    axs[0].set_title("Frame Rate")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(left=0)  # Start x-axis at 0

    # Jitter plot
    axs[1].plot(df["time_since_start"], df["input_jitter_ms"], label="Input Jitter (ms)", color="blue", alpha=0.7)
    axs[1].plot(df["time_since_start"], df["output_jitter_ms"], label="Output Jitter (ms)", color="green", alpha=0.7)
    axs[1].set_ylabel("Jitter (ms)")
    axs[1].set_title("Frame Timing Jitter (Rolling StdDev of Intervals)")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(left=0)  # Start x-axis at 0

    # Latency plot - show per-frame latency rather than cumulative
    axs[2].plot(df["time_since_start"], df["latency_ms"], label="Per-Frame Latency (ms)", color="red", alpha=0.5)
    axs[2].plot(df["time_since_start"], df["latency_ms_smooth"], label="Smoothed Latency (ms)", color="darkred", linewidth=2)
    axs[2].set_ylabel("Latency (ms)")
    axs[2].set_title("Frame Processing Latency")
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlim(left=0)  # Start x-axis at 0
    
    # Set y-axis limit to better visualize actual per-frame latency without accumulation
    max_latency = df["latency_ms"].quantile(0.99)  # Use 99th percentile to avoid outliers
    axs[2].set_ylim(0, max_latency * 1.1)  # Add 10% margin
    
    # Add x-axis label and make it visible
    fig.text(0.5, 0.04, 'Time (seconds)', ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07)  # Add space for x-axis label
    plt.savefig(csv_path.replace('.csv', '.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot frame timing metrics from CSV logs.")
    parser.add_argument(
        "--frame-logs",
        type=str,
        default="frame_logs.csv",
        help="Path to the frame timing CSV log file (default: frame_logs.csv)"
    )
    args = parser.parse_args()
    plot_frame_metrics(args.frame_logs)

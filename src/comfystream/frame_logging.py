# frame_logging.py
# Developed by @buffmcbighuge (Marco Tundo)

# You can generate graphs from the log file:
# python frame_logging.py --frame-logs frame_logs1.csv,frame_logs_2.csv,frame_logs_3.csv

import csv
import os
import time
from typing import Optional, Dict, Any, List, Tuple
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def log_frame_timing(
    frame_id: Optional[int],
    frame_received_time: Optional[float],
    frame_process_start_time: Optional[float],
    frame_processed_time: Optional[float],
    client_index: Optional[int] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    csv_path: str = "frame_logs.csv"
):
    """
    Log frame timing information to a CSV file with simplified metrics.
    Args:
        frame_id: The unique identifier for the frame
        frame_received_time: Timestamp when the frame was received by pipeline
        frame_process_start_time: Timestamp when processing began
        frame_processed_time: Timestamp when processing completed
        client_index: Index of the client that processed this frame
        additional_metadata: Any additional data to log
        csv_path: Path to the CSV file
    """
    # Calculate processing latency
    processing_latency = None
    if frame_process_start_time is not None and frame_processed_time is not None:
        processing_latency = (frame_processed_time - frame_process_start_time) * 1000

    # Convert additional metadata to string if present
    metadata_str = str(additional_metadata) if additional_metadata else None

    # Calculate absolute time for logging
    current_time = time.time()

    # Determine if this is an input-only frame or a processed frame
    is_processed = frame_process_start_time is not None and frame_processed_time is not None
    frame_type = "processed" if is_processed else "input"

    # Prepare data based on frame type
    if is_processed:
        # For processed frames, include all columns
        header = [
            "log_timestamp", "frame_id", "frame_type",
            "frame_received_time", "frame_process_start_time", "frame_processed_time",
            "processing_latency_ms", "client_index", "metadata"
        ]
        data = [
            current_time, frame_id, frame_type,
            frame_received_time, frame_process_start_time, frame_processed_time,
            processing_latency, client_index, metadata_str
        ]
    else:
        # For input frames, only include relevant columns (skip processing-related columns)
        header = [
            "log_timestamp", "frame_id", "frame_type",
            "frame_received_time", "client_index", "metadata"
        ]
        data = [
            current_time, frame_id, frame_type,
            frame_received_time, client_index, metadata_str
        ]

    file_exists = os.path.isfile(csv_path)
    file_empty = file_exists and os.path.getsize(csv_path) == 0

    # Use pandas to handle the CSV file, which handles mixed column formats better
    if not file_exists or file_empty:
        # If file doesn't exist or is empty, create a new one with the full header
        # This ensures the file always has all possible columns defined
        full_header = [
            "log_timestamp", "frame_id", "frame_type",
            "frame_received_time", "frame_process_start_time", "frame_processed_time",
            "processing_latency_ms", "client_index", "metadata"
        ]
        pd.DataFrame(columns=full_header).to_csv(csv_path, index=False)

    # Now append the data
    df = pd.DataFrame([dict(zip(header, data))])
    df.to_csv(csv_path, mode='a', header=False, index=False, columns=header)

def process_log_file(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Process a single log file and return the processed dataframes
    """
    if not os.path.isfile(csv_path):
        print(f"CSV file '{csv_path}' not found.")
        return None, None, None, 0

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"CSV file '{csv_path}' is empty.")
        return None, None, None, 0

    # Drop rows with missing essential times
    df = df.dropna(subset=["frame_id", "frame_received_time"])
    if df.empty:
        print("No valid timing data in CSV after dropping NA in essential time columns.")
        return None, None, None, 0

    # Sort by frame_id and calculate time since start
    df = df.sort_values("frame_id").reset_index(drop=True)
    stream_start_time = df["frame_received_time"].min()
    df["time_since_start"] = df["frame_received_time"] - stream_start_time

    # Separate input and processed frames based on frame_type column
    if "frame_type" in df.columns:
        input_df = df[df["frame_type"] == "input"].copy()
        processed_df = df[df["frame_type"] == "processed"].copy()
    else:
        # Backward compatibility - separate based on process timestamps
        input_df = df[df["frame_process_start_time"].isna()].copy()
        processed_df = df.dropna(subset=["frame_processed_time"]).copy()
    
    # Calculate time of processed frames relative to stream start
    if not processed_df.empty:
        processed_df.loc[:, "output_time_relative"] = processed_df["frame_processed_time"] - stream_start_time
    
    # Create a consistent timeline with fixed intervals based on overall activity
    max_input_time = df["time_since_start"].max() if not df["time_since_start"].empty else 0
    max_output_time = processed_df["output_time_relative"].max() if not processed_df.empty else 0
    
    max_time = max(max_input_time, max_output_time)
    time_range = np.arange(0, int(max_time) + 1)
    
    # Initialize FPS arrays for consistent timeline
    input_fps_counts = np.zeros(len(time_range))
    output_fps_counts = np.zeros(len(time_range))
    
    # Count frames in each 1-second interval
    for t_idx, t_sec in enumerate(time_range):
        # For input FPS, count input frames by received time
        if not input_df.empty:
            input_mask = (input_df["time_since_start"] >= t_sec) & (input_df["time_since_start"] < t_sec + 1)
            input_fps_counts[t_idx] = input_mask.sum()
        
        # For output FPS, count processed frames by processed time
        if not processed_df.empty:
            output_mask = (processed_df["output_time_relative"] >= t_sec) & (processed_df["output_time_relative"] < t_sec + 1)
            output_fps_counts[t_idx] = output_mask.sum()
    
    fps_df = pd.DataFrame({
        "time_bin": time_range,
        "input_fps": input_fps_counts,
        "output_fps": output_fps_counts
    })
    
    # Apply smoothing
    smoothing_window = 3
    fps_df["input_fps_smooth"] = fps_df["input_fps"].rolling(smoothing_window, min_periods=1).mean()
    fps_df["output_fps_smooth"] = fps_df["output_fps"].rolling(smoothing_window, min_periods=1).mean()
    
    # Calculate frame intervals for input and output frames separately
    # Only calculate intervals for the same frame type
    if not input_df.empty:
        input_df = input_df.sort_values("frame_received_time").reset_index(drop=True)
        input_df.loc[:, "input_interval_s"] = input_df["frame_received_time"].diff()
        input_df.loc[input_df["input_interval_s"] <= 0, "input_interval_s"] = np.nan
        input_df.loc[:, "input_time_bin"] = input_df["time_since_start"].astype(int)
    
    if not processed_df.empty:
        processed_df = processed_df.sort_values("frame_processed_time").reset_index(drop=True)
        processed_df.loc[:, "output_interval_s"] = processed_df["frame_processed_time"].diff()
        processed_df.loc[processed_df["output_interval_s"] <= 0, "output_interval_s"] = np.nan
        processed_df.loc[:, "output_time_bin"] = processed_df["output_time_relative"].astype(int)
    
    # Calculate jitter as the standard deviation of frame intervals in each time bin
    input_jitter = np.full(len(time_range), np.nan)
    output_jitter = np.full(len(time_range), np.nan)
    
    for t_idx, t_sec in enumerate(time_range):
        # Input jitter - variation in input frame arrival times
        if not input_df.empty:
            intervals = input_df.loc[input_df["input_time_bin"] == t_sec, "input_interval_s"]
            if len(intervals.dropna()) > 1:
                std_dev = intervals.std() * 1000  # Convert to ms
                input_jitter[t_idx] = std_dev
        
        # Output jitter - variation in processed frame completion times
        if not processed_df.empty:
            intervals = processed_df.loc[processed_df["output_time_bin"] == t_sec, "output_interval_s"]
            if len(intervals.dropna()) > 1:
                std_dev = intervals.std() * 1000  # Convert to ms
                output_jitter[t_idx] = std_dev
    
    jitter_df = pd.DataFrame({
        "time_bin": time_range,
        "input_jitter_ms": input_jitter,
        "output_jitter_ms": output_jitter
    })
    
    # Aggregate processing latency by time bin
    if not processed_df.empty:
        avg_latencies = processed_df.groupby("output_time_bin").agg({
            "processing_latency_ms": "mean"
        }).reset_index()
        
        latency_df = pd.DataFrame({"time_bin": time_range})
        latency_df = pd.merge(
            latency_df, 
            avg_latencies.rename(columns={"output_time_bin": "time_bin"}),
            on="time_bin", 
            how="left"
        )
    else:
        latency_df = pd.DataFrame({
            "time_bin": time_range,
            "processing_latency_ms": np.nan
        })
    
    return fps_df, jitter_df, latency_df, max_time

def plot_multiple_frame_metrics(csv_paths: List[str]):
    """
    Plot metrics from multiple log files on the same charts
    """
    if not csv_paths:
        print("No CSV files provided.")
        return
    
    # Process each log file
    all_data = []
    max_time_overall = 0
    
    for csv_path in csv_paths:
        fps_df, jitter_df, latency_df, max_time = process_log_file(csv_path)
        if fps_df is not None:
            all_data.append({
                'path': csv_path,
                'fps_df': fps_df,
                'jitter_df': jitter_df,
                'latency_df': latency_df,
                'max_time': max_time
            })
            max_time_overall = max(max_time_overall, max_time)
    
    if not all_data:
        print("No valid data found in any of the provided CSV files.")
        return
    
    # Create visualization with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Generate a list of distinct colors for multiple datasets
    # Use a subset of tab colors for better distinction
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i, data in enumerate(all_data):
        # Get colors for this dataset
        input_color = tab_colors[i % len(tab_colors)]
        output_color = tab_colors[(i + len(tab_colors)//2) % len(tab_colors)]
        
        # Extract the filename without path and extension for legend
        file_label = os.path.splitext(os.path.basename(data['path']))[0]
        
        # 1. FPS Plot
        axs[0].plot(
            data['fps_df']["time_bin"], 
            data['fps_df']["input_fps_smooth"], 
            label=f"Input FPS - {file_label}", 
            color=input_color, 
            linewidth=2
        )
        axs[0].plot(
            data['fps_df']["time_bin"], 
            data['fps_df']["output_fps_smooth"], 
            label=f"Output FPS - {file_label}", 
            color=output_color, 
            linewidth=2, 
            linestyle='--'
        )
        
        # 2. Jitter Plot
        axs[1].plot(
            data['jitter_df']["time_bin"], 
            data['jitter_df']["input_jitter_ms"], 
            label=f"Input Jitter - {file_label}", 
            color=input_color, 
            alpha=0.7
        )
        axs[1].plot(
            data['jitter_df']["time_bin"], 
            data['jitter_df']["output_jitter_ms"], 
            label=f"Output Jitter - {file_label}", 
            color=output_color, 
            alpha=0.7, 
            linestyle='--'
        )
        
        # 3. Processing Latency Plot
        axs[2].plot(
            data['latency_df']["time_bin"], 
            data['latency_df']["processing_latency_ms"], 
            label=f"Processing Latency - {file_label}", 
            color=output_color, 
            alpha=0.7
        )
    
    # Configure axes and labels
    axs[0].set_ylabel("FPS")
    axs[0].set_title("Frame Rate")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(left=0, right=max_time_overall if max_time_overall > 0 else 1) 
    axs[0].set_ylim(bottom=0)
    
    axs[1].set_ylabel("Jitter (ms)")
    axs[1].set_title("Frame Timing Jitter")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(left=0, right=max_time_overall if max_time_overall > 0 else 1)
    
    axs[2].set_ylabel("Latency (ms)")
    axs[2].set_title("Processing Latency")
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlim(left=0, right=max_time_overall if max_time_overall > 0 else 1)
    
    # Add x-axis label
    fig.text(0.5, 0.04, 'Time (seconds)', ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07)
    
    # Save combined plot
    output_filename = "combined_frame_logs.png"
    plt.savefig(output_filename)
    print(f"Combined plot saved as {output_filename}")
    plt.show()

def plot_frame_metrics(csv_path: str = "frame_logs.csv"):
    """
    Plot metrics from a single log file for backward compatibility
    """
    # For single files, just call the multiple processing function with a list of one item
    plot_multiple_frame_metrics([csv_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot frame timing metrics from CSV logs.")
    parser.add_argument(
        "--frame-logs",
        type=str,
        default="frame_logs.csv",
        help="Path to the frame timing CSV log file(s) (comma-separated for multiple files)"
    )
    args = parser.parse_args()
    
    # Check if multiple files are specified
    csv_paths = [path.strip() for path in args.frame_logs.split(',')]
    
    if len(csv_paths) > 1:
        plot_multiple_frame_metrics(csv_paths)
    else:
        plot_frame_metrics(csv_paths[0])

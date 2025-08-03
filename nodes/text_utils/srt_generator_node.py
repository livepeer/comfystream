"""
SRT subtitle generation node for ComfyStream.
Based on project-transcript SRT generation capabilities.
"""

import logging
from datetime import timedelta
from typing import List, Dict, Any, Union
import json

logger = logging.getLogger(__name__)


class SRTGeneratorNode:
    """
    Generate SRT subtitle files from transcription with timing information.
    Compatible with AudioTranscriptionNode output when word timestamps are enabled.
    """
    
    CATEGORY = "text_utils"
    RETURN_TYPES = ("STRING",)
    
    def __init__(self):
        self.subtitle_counter = 1
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcription_data": ("STRING", {
                    "tooltip": "Transcription text or JSON with timing data from AudioTranscriptionNode"
                }),
                "segment_start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 86400.0,  # 24 hours max
                    "step": 0.001,
                    "tooltip": "Start time of this segment in seconds (for absolute timing)"
                }),
                "segment_duration": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 60.0,
                    "step": 0.1,
                    "tooltip": "Duration of this segment in seconds"
                }),
                "use_absolute_time": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use absolute timing (True) or segment-relative timing (False)"
                }),
                "minimum_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Minimum subtitle duration in seconds"
                })
            }
        }
    
    @classmethod
    def RETURN_NAMES(cls):
        return ("srt_content",)
    
    FUNCTION = "generate_srt"
    
    @classmethod
    def IS_CHANGED(cls):
        return float("nan")
    
    def generate_srt(self, 
                    transcription_data: str,
                    segment_start_time: float = 0.0,
                    segment_duration: float = 3.0,
                    use_absolute_time: bool = False,
                    minimum_duration: float = 1.0) -> tuple:
        """
        Generate SRT subtitle content from transcription data.
        
        Args:
            transcription_data: Text transcription or JSON with timing data
            segment_start_time: Start time of this segment (for absolute timing)
            segment_duration: Duration of this segment
            use_absolute_time: Whether to use absolute or segment-relative timing
            minimum_duration: Minimum subtitle duration
            
        Returns:
            Tuple containing SRT formatted content
        """
        try:
            if not transcription_data or not transcription_data.strip() or len(transcription_data.strip()) < 5:
                logger.debug(f"Empty or minimal transcription data: '{transcription_data}'")
                return ("",)
            
            # Pass through warmup sentinel values (they'll be filtered later in the pipeline)
            if "__WARMUP_SENTINEL__" in transcription_data:
                logger.debug("Passing through warmup sentinel for pipeline detection")
                return (transcription_data,)
            
            # Try to parse as JSON first (if AudioTranscriptionNode supports structured output)
            segments = self._parse_transcription_data(transcription_data)
            
            if not segments:
                logger.debug("No valid segments found in transcription data")
                return ("",)
            
            # Generate SRT content
            srt_content = self._generate_srt_from_segments(
                segments, 
                segment_start_time,
                segment_duration,
                use_absolute_time,
                minimum_duration
            )
            
            logger.debug(f"Generated SRT with {len(segments)} segments")
            return (srt_content,)
            
        except Exception as e:
            logger.error(f"Error generating SRT: {e}")
            return ("",)
    
    def _parse_transcription_data(self, data: str) -> List[Dict[str, Any]]:
        """
        Parse transcription data into segments.
        Enhanced to handle AudioTranscriptionNode JSON output formats.
        """
        segments = []
        
        try:
            # Try parsing as JSON first
            parsed_data = json.loads(data)
            
            if isinstance(parsed_data, list):
                # Handle both segment-level and word-level JSON arrays
                for item in parsed_data:
                    if isinstance(item, dict):
                        # Check if it's word-level data (has 'word' field) or segment-level (has 'text' field)
                        if 'word' in item:
                            # Word-level JSON from AudioTranscriptionNode (json_words format)
                            segments.append({
                                'start': item.get('start', 0.0),
                                'end': item.get('end', 1.0),
                                'text': item.get('word', '').strip()
                            })
                        elif 'text' in item:
                            # Segment-level JSON from AudioTranscriptionNode (json_segments format)
                            segments.append({
                                'start': item.get('start', 0.0),
                                'end': item.get('end', 1.0),
                                'text': item.get('text', '').strip()
                            })
                            
            elif isinstance(parsed_data, dict):
                # Single segment or word
                if 'word' in parsed_data:
                    segments.append({
                        'start': parsed_data.get('start', 0.0),
                        'end': parsed_data.get('end', 1.0),
                        'text': parsed_data.get('word', '').strip()
                    })
                elif 'text' in parsed_data:
                    segments.append({
                        'start': parsed_data.get('start', 0.0),
                        'end': parsed_data.get('end', 1.0),
                        'text': parsed_data.get('text', '').strip()
                    })
                    
        except (json.JSONDecodeError, TypeError):
            # Fallback to plain text - create a single segment
            if data.strip():
                segments.append({
                    'start': 0.0,
                    'end': 3.0,  # Default 3-second duration
                    'text': data.strip()
                })
        
        return [seg for seg in segments if seg['text']]  # Filter empty text
    
    def _generate_srt_from_segments(self, 
                                   segments: List[Dict[str, Any]],
                                   segment_start_time: float,
                                   segment_duration: float,
                                   use_absolute_time: bool,
                                   minimum_duration: float) -> str:
        """
        Generate SRT content from parsed segments.
        """
        srt_lines = []
        
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            if not text:
                continue
            
            # Calculate timing
            if use_absolute_time:
                # Absolute timing: add segment start time
                start_time = segment_start_time + segment['start']
                end_time = segment_start_time + segment['end']
            else:
                # Segment-relative timing
                start_time = segment['start']
                end_time = segment['end']
            
            # Ensure minimum duration
            if end_time - start_time < minimum_duration:
                end_time = start_time + minimum_duration
            
            # Format timing
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            # Add SRT entry
            srt_lines.append(str(self.subtitle_counter))
            srt_lines.append(f"{start_srt} --> {end_srt}")
            srt_lines.append(text)
            srt_lines.append("")  # Blank line
            
            self.subtitle_counter += 1
        
        return "\n".join(srt_lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT time format (HH:MM:SS,mmm).
        Compatible with project-transcript format.
        """
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        milliseconds = int((seconds - total_seconds) * 1000)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


# Node registration is handled in __init__.py
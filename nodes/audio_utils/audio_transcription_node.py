"""
Real-time Audio Transcription Node for ComfyUI.

This node buffers audio segments and performs transcription using faster-whisper,
with controlled output timing to prevent message flooding.
"""

import asyncio
import json
import logging
import time
import tempfile
import os
import numpy as np
from typing import Optional, List, Deque, Dict, Any
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
from queue import Queue, Empty
from faster_whisper import WhisperModel

from comfystream import tensor_cache

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Represents a transcribed segment with timing information."""
    start: float  # Start time in seconds (buffer-relative)
    end: float    # End time in seconds (buffer-relative)
    text: str     # Transcribed text
    confidence: float = 0.0  # Confidence score


class AudioTranscriptionNode:
    """
    Real-time audio transcription node that buffers audio and outputs 
    transcribed text on a controlled schedule to prevent message flooding.
    """
    
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("STRING",)
    
    def __init__(self):
        # Audio buffering
        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.buffer_duration = 0.0  # Duration in seconds
        self.sample_rate = None
        self.buffer_samples = None
        
        # Whisper model (shared across instances)
        self._whisper_model = None
        self._model_lock = threading.Lock()
        
        # Transcription queue for controlled output
        self.transcription_queue: Deque[str] = deque(maxlen=50)
        self.last_output_time = 0.0
        
        # Processing state
        self.total_audio_processed = 0.0  # Total audio time processed
        self.transcription_count = 0
        
        # Warmup state - only generate sentinels during initial warmup phase
        self.warmup_phase = True
        self.successful_transcriptions = 0
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("WAVEFORM",),  # Input from LoadAudioTensor
                "transcription_interval": ("FLOAT", {
                    "default": 4.0, 
                    "min": 1.0, 
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Minimum seconds between transcription outputs"
                }),
                "buffer_duration": ("FLOAT", {
                    "default": 4.0,
                    "min": 2.0, 
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "Audio buffer duration in seconds for transcription"
                }),
                "whisper_model": (["tiny", "base", "small", "medium", "large-v2"], {
                    "default": "base",
                    "tooltip": "Whisper model size (larger = more accurate but slower)"
                }),
                "language": (["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"], {
                    "default": "auto",
                    "tooltip": "Language for transcription (auto = auto-detect)"
                }),
                "enable_vad": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Voice Activity Detection to filter silence"
                })
            },
            "optional": {
                "output_format": (["text", "json_segments", "json_words"], {
                    "default": "json_segments",
                    "tooltip": "Output format: text (simple), json_segments (with timing), json_words (word-level timing)"
                })
            }
        }
    
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")
    


    def _initialize_whisper_model(self, model_size: str):
        """Initialize the Whisper model if not already loaded."""
        with self._model_lock:
            # Load model if not already loaded or if different size requested
            if (self._whisper_model is None or 
                getattr(self._whisper_model, 'model_size_name', '') != model_size):
                
                logger.info(f"Loading Whisper model: {model_size}")
                self._load_whisper_model_now(model_size)



    def _load_whisper_model_now(self, model_size: str):
        """Load the Whisper model using faster-whisper's automatic download."""
        # Use CPU for compatibility, can be changed to CUDA if available
        device = "cpu"
        try:
            # Try CUDA first if available
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                compute_type = "int8"
        except:
            compute_type = "int8"
        
        # Let faster-whisper handle model downloading and caching automatically
        # This ensures we get the correct file structure (vocabulary.txt, etc.)
        logger.info(f"Loading Whisper model via faster-whisper: {model_size}")
        
        self._whisper_model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type
        )
        self._whisper_model.model_size_name = model_size
        logger.info(f"Whisper model '{model_size}' loaded successfully on {device}")

    def _ensure_model_loaded(self, model_size: str):
        """Ensure the Whisper model is loaded and ready for transcription."""
        with self._model_lock:
            # Check if we need to load or switch models
            if (self._whisper_model is None or 
                getattr(self._whisper_model, 'model_size_name', '') != model_size):
                
                logger.info(f"Loading Whisper model: {model_size}")
                self._load_whisper_model_now(model_size)
    

    
    def _buffer_audio(self, audio_input: np.ndarray, sample_rate: int, buffer_duration: float):
        """
        Efficiently buffer audio data optimized for Whisper transcription.
        
        This is the primary buffering system - no duplicate buffering from LoadAudioTensor.
        """
        # Initialize buffer parameters
        if self.sample_rate is None:
            self.sample_rate = sample_rate
            self.buffer_samples = int(self.sample_rate * buffer_duration)
            logger.info(f"Initialized Whisper-optimized audio buffer: {buffer_duration}s at {sample_rate}Hz = {self.buffer_samples} samples")
        
        # Skip empty frames (from streaming loader)
        if audio_input is None or audio_input.size == 0:
            return False
        
        # Ensure audio is in the right format for Whisper
        audio_input = self._normalize_audio_for_whisper(audio_input)
        
        # Add to buffer - concatenate efficiently
        if self.audio_buffer.size == 0:
            self.audio_buffer = audio_input.copy()
        else:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
        
        # Update buffer duration
        self.buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        # Check if we have enough audio for transcription
        ready = len(self.audio_buffer) >= self.buffer_samples
        
        if ready:
            logger.debug(f"Audio buffer ready for transcription: {self.buffer_duration:.2f}s ({len(self.audio_buffer)} samples)")
        
        return ready
    
    def _normalize_audio_for_whisper(self, audio_input: np.ndarray) -> np.ndarray:
        """
        Normalize audio specifically for Whisper requirements.
        
        Whisper expects:
        - 16-bit PCM audio (int16)
        - Mono channel
        - 16 kHz sample rate (handled by LoadAudioTensorStream)
        """
        if audio_input is None or audio_input.size == 0:
            return np.array([], dtype=np.int16)
        
        # Ensure numpy array
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input)
        
        # Handle multi-channel audio (take first channel for transcription)
        if audio_input.ndim > 1:
            if audio_input.shape[1] > audio_input.shape[0]:
                # Shape is (samples, channels) - take first channel
                audio_input = audio_input[:, 0]
            else:
                # Shape is (channels, samples) - take first channel  
                audio_input = audio_input[0, :]
        
        # Convert to int16 format for Whisper
        if audio_input.dtype != np.int16:
            if audio_input.dtype in [np.float32, np.float64]:
                # Convert from float [-1, 1] to int16 [-32768, 32767]
                audio_input = np.clip(audio_input, -1.0, 1.0)
                audio_input = (audio_input * 32767).astype(np.int16)
            else:
                audio_input = audio_input.astype(np.int16)
        
        return audio_input
    
    def _transcribe_audio_buffer(self, model_size: str, language: str, enable_vad: bool, output_format: str = "json_segments") -> Optional[str]:
        """Transcribe the current audio buffer and return combined text."""
        if len(self.audio_buffer) < self.buffer_samples:
            return None
        
        try:
            # Ensure model is loaded (handles deferred loading from warmup)
            self._ensure_model_loaded(model_size)
            
            # Extract audio chunk for transcription
            audio_chunk = self.audio_buffer[:self.buffer_samples]
            
            # Save audio to temporary WAV file for whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Write WAV file using scipy.io.wavfile
                try:
                    from scipy.io.wavfile import write
                    write(temp_path, self.sample_rate, audio_chunk)
                except ImportError:
                    # Fallback to manual WAV writing if scipy not available
                    self._write_wav_file(temp_path, audio_chunk, self.sample_rate)
                
                # Transcribe using whisper
                language_code = None if language == "auto" else language
                segments, info = self._whisper_model.transcribe(
                    temp_path,
                                    language=language_code,
                word_timestamps=True,  # Enable for SRT generation
                vad_filter=enable_vad,
                    beam_size=1,  # Faster transcription
                    best_of=1     # Faster transcription
                )
                
                # Format output based on requested format
                if output_format == "text":
                    # Simple text output (original behavior)
                    transcribed_texts = []
                    for segment in segments:
                        if segment.text.strip():
                            transcribed_texts.append(segment.text.strip())
                    result = " ".join(transcribed_texts)
                    
                elif output_format == "json_segments":
                    # Segment-level JSON output (like project-transcript)
                    segments_data = []
                    for segment in segments:
                        if segment.text.strip():
                            segments_data.append({
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text.strip()
                            })
                    result = json.dumps(segments_data, ensure_ascii=False)
                    
                elif output_format == "json_words":
                    # Word-level JSON output (detailed timing)
                    words_data = []
                    for segment in segments:
                        if hasattr(segment, 'words') and segment.words:
                            for word in segment.words:
                                if word.word.strip():
                                    words_data.append({
                                        "start": word.start,
                                        "end": word.end, 
                                        "word": word.word.strip(),
                                        "probability": getattr(word, 'probability', 1.0)
                                    })
                        elif segment.text.strip():
                            # Fallback if word-level timing not available
                            words_data.append({
                                "start": segment.start,
                                "end": segment.end,
                                "word": segment.text.strip(),
                                "probability": 1.0
                            })
                    result = json.dumps(words_data, ensure_ascii=False)
                else:
                    result = None
                
                # Only generate sentinel values during warmup phase
                if not result or (output_format in ["json_segments", "json_words"] and result == "[]"):
                    if self.warmup_phase:
                        # Return sentinel value to indicate the pipeline is working during warmup
                        if output_format == "text":
                            result = "__WARMUP_SENTINEL__"
                        elif output_format == "json_segments":
                            result = json.dumps([{"start": 0.0, "end": 1.0, "text": "__WARMUP_SENTINEL__"}])
                        elif output_format == "json_words":
                            result = json.dumps([{"start": 0.0, "end": 1.0, "word": "__WARMUP_SENTINEL__", "probability": 1.0}])
                        logger.debug(f"No transcription produced during warmup, returning sentinel value")
                    else:
                        # During normal operation, return None for empty transcriptions
                        result = ""  # Return empty string for ComfyUI compatibility 
                        logger.debug(f"No transcription produced during normal operation, returning empty")
                
                # Track successful transcriptions and exit warmup phase
                if result and not ("__WARMUP_SENTINEL__" in result):
                    self.successful_transcriptions += 1
                    if self.successful_transcriptions >= 2:
                        self.warmup_phase = False
                        logger.debug("Exited warmup phase - future empty transcriptions will return None")
                
                # Update processing metrics
                self.total_audio_processed += self.buffer_samples / self.sample_rate
                self.transcription_count += 1
                
                # Advance buffer (keep some overlap for context)
                overlap_samples = self.buffer_samples // 4  # 25% overlap
                advance_samples = self.buffer_samples - overlap_samples
                self.audio_buffer = self.audio_buffer[advance_samples:]
                
                if result:
                    logger.debug(f"Transcribed chunk {self.transcription_count} ({output_format}): '{str(result)[:50]}...' ({len(str(result))} chars)")
                else:
                    logger.debug(f"Transcribed chunk {self.transcription_count} ({output_format}): No result")
                
                return result
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            # Still advance buffer to prevent getting stuck
            self.audio_buffer = self.audio_buffer[self.buffer_samples // 2:]
            return None
    
    def _write_wav_file(self, filename: str, audio_data: np.ndarray, sample_rate: int):
        """Write WAV file manually if scipy is not available."""
        import struct
        import wave
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    def _should_output_transcription(self, transcription_interval: float) -> bool:
        """Check if we should output transcription based on queue or timing."""
        # For real-time streaming, prioritize queue content
        if len(self.transcription_queue) > 0:
            # logger.debug(f"Queue has {len(self.transcription_queue)} items - outputting immediately")  # Too frequent
            return True
            
        # Fallback to timing interval only if queue is empty
        current_time = time.time()
        
        # During warmup phase, use shorter intervals to speed up warmup detection
        if self.warmup_phase:
            effective_interval = min(transcription_interval, 2.0)  # Max 2 seconds during warmup
            logger.debug(f"Warmup phase - using {effective_interval}s interval instead of {transcription_interval}s")
        else:
            effective_interval = transcription_interval
            
        return (current_time - self.last_output_time) >= effective_interval
    
    def _get_queued_transcription(self) -> Optional[str]:
        """Get transcription from queue if available."""
        try:
            return self.transcription_queue.popleft()
        except IndexError:
            return None
    
    def execute(self, audio, transcription_interval=8.0, buffer_duration=8.0, 
                whisper_model="base", language="auto", enable_vad=True, output_format="json_segments"):
        """
        Execute transcription on streaming audio input.
        
        Args:
            audio: Audio input from LoadAudioTensorStream - tuple of (audio_data, sample_rate)
            transcription_interval: Minimum seconds between outputs (prevents message flooding)
            buffer_duration: Audio buffer duration for optimal Whisper transcription
            whisper_model: Whisper model size (tiny/base/small/medium/large-v2)
            language: Language for transcription (auto for auto-detection)
            enable_vad: Enable voice activity detection to filter silence
            output_format: Output format (text/json_segments/json_words)
            
        Returns:
            Tuple containing transcribed text (empty string if not ready to output)
        """
        try:
            # Parse audio input - expect (audio_data, sample_rate) from LoadAudioTensorStream
            if isinstance(audio, tuple) and len(audio) == 2:
                audio_data, sample_rate = audio
            elif hasattr(audio, 'shape'):
                # Fallback for direct numpy array (backward compatibility)
                audio_data = audio
                sample_rate = 16000  # Default Whisper-optimized rate
                logger.debug("Using fallback audio format detection")
            else:
                logger.warning(f"Unexpected audio format: {type(audio)}, returning empty")
                return ("",)
            
            # Validate sample rate for Whisper optimization
            if sample_rate != 16000:
                logger.warning(f"Non-optimal sample rate {sample_rate}Hz for Whisper (16kHz recommended)")
            

            
            # Buffer the audio (primary buffering system - no duplicate buffering)
            ready_for_transcription = self._buffer_audio(audio_data, sample_rate, buffer_duration)
            
            # Transcribe if buffer has enough audio
            if ready_for_transcription:
                transcription = self._transcribe_audio_buffer(whisper_model, language, enable_vad, output_format)
                if transcription and transcription.strip() and len(transcription.strip()) > 5:
                    # Add to queue for controlled output timing (only substantial content or warmup sentinels)
                    self._queue_transcription(transcription)
                    is_sentinel = "__WARMUP_SENTINEL__" in transcription
                    if is_sentinel:
                        logger.debug(f"Queued warmup sentinel ({output_format})")
                    else:
                        logger.debug(f"Queued transcription ({output_format}): '{transcription[:50]}...' (length: {len(transcription)})")
                elif transcription and transcription != "":
                    logger.debug(f"Skipping minimal transcription: '{transcription}' (too short)")
                elif transcription == "":
                    logger.debug(f"Empty transcription result (post-warmup silence)")
                else:
                    logger.debug(f"No transcription result returned")
            
            # Output transcriptions immediately when available (real-time streaming)
            if self._should_output_transcription(transcription_interval):
                output_text = self._get_queued_transcription()
                if output_text and output_text.strip():
                    self.last_output_time = time.time()
                    
                    # Log differently for sentinel vs regular transcription
                    if "__WARMUP_SENTINEL__" in output_text:
                        logger.info(f"Outputting warmup sentinel immediately ({output_format}, {len(output_text)} chars): '{output_text[:100]}...'")
                    else:
                        logger.info(f"Outputting transcription immediately ({output_format}, {len(output_text)} chars): '{output_text[:100]}...' (queue: {len(self.transcription_queue)})")
                    
                    return (output_text,)
            
            # Return empty string if not ready to output (SaveTextTensor will filter empty content)
            return ("",)
            
        except Exception as e:
            logger.error(f"Error in audio transcription execute: {e}")
            return ("",)
    
    def _queue_transcription(self, transcription: str):
        """Safely queue transcription for controlled output."""
        try:
            self.transcription_queue.append(transcription)
        except:
            # Queue full, remove oldest and add new
            try:
                self.transcription_queue.popleft()
                self.transcription_queue.append(transcription)
                logger.debug("Transcription queue full, replaced oldest entry")
            except:
                logger.warning("Failed to queue transcription")


# Register the node
NODE_CLASS_MAPPINGS = {
    "AudioTranscriptionNode": AudioTranscriptionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTranscriptionNode": "Audio Transcription (Real-time)"
}
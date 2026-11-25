"""
Voice module: Real-time audio processing pipeline.
Handles audio chunking, VAD, and coordination between STT and TTS.
"""

import io
import os
import time
import logging
import threading
import numpy as np
from typing import Optional, Callable, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Simple energy-based Voice Activity Detection.
    Fallback when Silero VAD is unavailable.
    """
    
    def __init__(self, threshold: float = 0.02, min_speech_ms: int = 300):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
    
    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Detect if audio chunk contains speech."""
        if audio is None or len(audio) == 0:
            return False
        
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        return rms > self.threshold
    
    def get_speech_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        frame_ms: int = 30
    ) -> list:
        """Get list of (start, end) speech segments in samples."""
        frame_size = int(sample_rate * frame_ms / 1000)
        segments = []
        in_speech = False
        start = 0
        
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break
            
            is_speech_frame = self.is_speech(frame, sample_rate)
            
            if is_speech_frame and not in_speech:
                start = i
                in_speech = True
            elif not is_speech_frame and in_speech:
                min_samples = int(sample_rate * self.min_speech_ms / 1000)
                if i - start >= min_samples:
                    segments.append((start, i))
                in_speech = False
        
        if in_speech:
            segments.append((start, len(audio)))
        
        return segments


class AudioProcessor:
    """
    Coordinates the audio processing pipeline:
    microphone → VAD → STT → LLM → TTS → speaker
    """
    
    def __init__(self):
        self.vad = VoiceActivityDetector()
        self.sample_rate = 16000
        self.is_processing = False
        self._lock = threading.Lock()
        self._last_process_time = 0
        self._min_process_interval = 0.5  # Prevent duplicate processing
    
    def process_audio_chunk(
        self,
        audio_data,
        transcribe_fn: Callable,
        response_fn: Callable,
        synthesize_fn: Callable
    ) -> Optional[Tuple[str, str, np.ndarray, int]]:
        """
        Process a single audio chunk through the full pipeline.
        
        Args:
            audio_data: Raw audio (numpy array or tuple from Gradio)
            transcribe_fn: Function to transcribe audio
            response_fn: Function to generate LLM response
            synthesize_fn: Function to synthesize TTS
        
        Returns:
            (transcript, response, audio_array, sample_rate) or None
        """
        # Prevent duplicate processing
        now = time.time()
        if now - self._last_process_time < self._min_process_interval:
            return None
        
        with self._lock:
            if self.is_processing:
                logger.debug("Already processing audio, skipping")
                return None
            self.is_processing = True
        
        try:
            # Parse audio input (Gradio returns (sample_rate, numpy_array))
            audio_array, sr = self._parse_audio_input(audio_data)
            
            if audio_array is None or len(audio_array) < sr * 0.3:  # Min 300ms
                return None
            
            # VAD check
            if not self.vad.is_speech(audio_array, sr):
                logger.debug("No speech detected in audio chunk")
                return None
            
            # STT
            transcript, detected_lang, confidence = transcribe_fn(audio_array, sr)
            
            if not transcript or len(transcript.strip()) < 2:
                logger.debug("Empty or very short transcript")
                return None
            
            logger.info(f"Transcript: '{transcript}' (lang={detected_lang}, conf={confidence:.2f})")
            
            # LLM Response
            response_text = response_fn(transcript, detected_lang)
            
            if not response_text:
                return None
            
            # TTS
            tts_result = synthesize_fn(response_text, detected_lang)
            
            self._last_process_time = time.time()
            
            if tts_result:
                audio_out, sr_out = tts_result
                return transcript, response_text, audio_out, sr_out
            else:
                return transcript, response_text, None, None
                
        except Exception as e:
            logger.error(f"Audio processing pipeline error: {e}")
            return None
        finally:
            with self._lock:
                self.is_processing = False
    
    def _parse_audio_input(self, audio_data) -> Tuple[Optional[np.ndarray], int]:
        """Parse various audio input formats."""
        if audio_data is None:
            return None, self.sample_rate
        
        # Gradio format: (sample_rate, numpy_array)
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, audio = audio_data
            if isinstance(audio, np.ndarray):
                # Convert to mono float32
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                if audio.dtype != np.float32:
                    # Gradio often gives int16
                    if audio.dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0
                    else:
                        audio = audio.astype(np.float32)
                return audio, int(sr)
        
        # Raw numpy array
        if isinstance(audio_data, np.ndarray):
            audio = audio_data.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return audio, self.sample_rate
        
        return None, self.sample_rate


def numpy_to_gradio_audio(audio: np.ndarray, sample_rate: int) -> Tuple[int, np.ndarray]:
    """Convert numpy float32 audio to Gradio audio format (int16)."""
    audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    return sample_rate, audio_int16

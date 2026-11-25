"""
Voice module: Speech-to-Text using Faster-Whisper.
CPU-optimized, int8 quantization, multilingual support.
"""

import io
import os
import time
import logging
import tempfile
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Global model instance (lazy-loaded, singleton)
_whisper_model = None
_model_lock = None


def _get_lock():
    """Get or create thread lock for model access."""
    global _model_lock
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()
    return _model_lock


def load_whisper_model(model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
    """
    Load Faster-Whisper model (singleton pattern).
    
    Args:
        model_size: "tiny", "base", "small", "medium"
        device: "cpu" or "cuda"
        compute_type: "int8", "float16", "float32"
    """
    global _whisper_model
    
    with _get_lock():
        if _whisper_model is not None:
            return _whisper_model
        
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Whisper model: {model_size} on {device} with {compute_type}")
            start = time.time()
            
            _whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=os.cpu_count() or 4,
                num_workers=1
            )
            
            elapsed = time.time() - start
            logger.info(f"Whisper model loaded in {elapsed:.2f}s")
            return _whisper_model
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise


def transcribe_audio(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Tuple[str, str, float]:
    """
    Transcribe audio data using local Faster-Whisper.
    """
    # Validate audio
    if audio_data is None or len(audio_data) == 0:
        return "", "en", 0.0
    
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    if sample_rate != 16000:
        audio_data = _resample(audio_data, sample_rate, 16000)
    
    if _is_silence(audio_data):
        logger.debug("Audio is silence, skipping transcription")
        return "", "en", 0.0

    model = load_whisper_model(
        model_size=os.getenv("WHISPER_MODEL", "small"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    )
    
    if model is None:
        raise RuntimeError("Whisper model not loaded")
    
    try:
        start_time = time.time()
        
        segments, info = model.transcribe(
            audio_data,
            language=language,
            task=task,
            beam_size=3,
            best_of=3,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100
            )
        )
        
        # Collect segments
        texts = []
        total_confidence = 0.0
        segment_count = 0
        
        for segment in segments:
            if segment.text.strip():
                texts.append(segment.text.strip())
                # Approximate confidence from log probability
                if hasattr(segment, 'avg_logprob'):
                    total_confidence += min(1.0, max(0.0, (segment.avg_logprob + 1.0)))
                    segment_count += 1
        
        transcript = " ".join(texts).strip()
        detected_lang = info.language if info else "en"
        avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.8
        
        elapsed = time.time() - start_time
        logger.info(f"STT: '{transcript[:50]}...' | Lang: {detected_lang} | Time: {elapsed:.2f}s")
        
        return transcript, detected_lang, avg_confidence
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise


def transcribe_audio_file(
    file_path: str,
    language: Optional[str] = None
) -> Tuple[str, str, float]:
    """Transcribe an audio file."""
    import soundfile as sf
    
    audio_data, sample_rate = sf.read(file_path, dtype='float32')
    
    # Convert stereo to mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    return transcribe_audio(audio_data, sample_rate, language)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate using scipy."""
    try:
        from scipy.signal import resample_poly
        from math import gcd
        
        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
    except Exception:
        # Simple linear interpolation fallback
        target_len = int(len(audio) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)


def _is_silence(audio: np.ndarray, threshold: float = 0.005) -> bool:
    """Check if audio is effectively silence."""
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < threshold


def detect_language_from_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Detect language from audio without full transcription."""
    model = load_whisper_model()
    if model is None:
        return "en"
    
    try:
        # Normalize to float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if sample_rate != 16000:
            audio_data = _resample(audio_data, sample_rate, 16000)
        
        # Use first 30 seconds for detection
        audio_30s = audio_data[:16000 * 30]
        
        _, info = model.transcribe(audio_30s, task="transcribe", beam_size=1)
        return info.language if info else "en"
        
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"


def check_stt_health() -> dict:
    """Check if STT model is available."""
    try:
        model = load_whisper_model()
        return {
            "status": "healthy" if model else "unhealthy",
            "model": os.getenv("WHISPER_MODEL", "small"),
            "device": os.getenv("WHISPER_DEVICE", "cpu"),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

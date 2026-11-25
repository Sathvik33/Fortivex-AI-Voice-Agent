"""
Voice module: Text-to-Speech using Piper TTS.
Local inference, CPU-optimized, multilingual support.
Downloads voice models on first use.
"""

import os
import io
import time
import logging
import threading
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("PIPER_MODELS_DIR", "./models"))
MODELS_DIR.mkdir(exist_ok=True)

# Voice model configurations
VOICE_CONFIGS = {
    "en": {
        "name": "en_US-lessac-medium",
        "url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/",
        "onnx": "en_US-lessac-medium.onnx",
        "json": "en_US-lessac-medium.onnx.json",
    },
    "hi": {
        # Piper doesn't have a Hindi voice - fall back to English
        "name": "en_US-lessac-medium",
        "url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/",
        "onnx": "en_US-lessac-medium.onnx",
        "json": "en_US-lessac-medium.onnx.json",
    }
}

# Global TTS voice instances
_voices: dict = {}
_tts_lock = threading.Lock()
_piper_available: Optional[bool] = None


def _check_piper_available() -> bool:
    """Check if piper-tts Python package is available."""
    global _piper_available
    if _piper_available is not None:
        return _piper_available
    try:
        from piper.voice import PiperVoice
        _piper_available = True
    except ImportError:
        _piper_available = False
        logger.warning("piper-tts not available, will use gTTS fallback")
    return _piper_available


def download_voice_model(lang: str = "en") -> Tuple[Optional[Path], Optional[Path]]:
    """
    Download Piper voice model if not already present.
    Returns (onnx_path, json_path) or (None, None) on failure.
    """
    config = VOICE_CONFIGS.get(lang, VOICE_CONFIGS["en"])
    onnx_path = MODELS_DIR / config["onnx"]
    json_path = MODELS_DIR / config["json"]
    
    if onnx_path.exists() and json_path.exists():
        logger.info(f"Voice model already downloaded: {config['name']}")
        return onnx_path, json_path
    
    try:
        import requests
        logger.info(f"Downloading voice model: {config['name']}")
        
        for filename, path in [(config["onnx"], onnx_path), (config["json"], json_path)]:
            url = config["url_base"] + filename
            logger.info(f"Downloading {url}")
            
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {path}")
        
        return onnx_path, json_path
        
    except Exception as e:
        logger.error(f"Failed to download voice model: {e}")
        return None, None


def load_voice(lang: str = "en"):
    """Load Piper voice model (singleton pattern)."""
    global _voices
    
    with _tts_lock:
        if lang in _voices:
            return _voices[lang]
        
        if not _check_piper_available():
            return None
        
        try:
            from piper.voice import PiperVoice
            
            onnx_path, json_path = download_voice_model(lang)
            if onnx_path is None:
                logger.error(f"Could not load voice for language: {lang}")
                return None
            
            start = time.time()
            voice = PiperVoice.load(str(onnx_path), config_path=str(json_path), use_cuda=False)
            elapsed = time.time() - start
            logger.info(f"Piper voice loaded in {elapsed:.2f}s for lang={lang}")
            
            _voices[lang] = voice
            return voice
            
        except Exception as e:
            logger.error(f"Failed to load Piper voice: {e}")
            return None


def synthesize_speech(
    text: str,
    language: str = "en",
    speaking_rate: float = 1.0
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Synthesize text to speech.
    - Hindi / Hinglish: Edge-TTS neural Indian voices (hi-IN-SwaraNeural / en-IN-NeerjaNeural)
    - English: Local Piper TTS (en_US-lessac-medium) -> Edge-TTS / gTTS fallback
    """
    if not text or not text.strip():
        return None
    
    text = _clean_text_for_tts(text, language)
    start_time = time.time()
    result = None

    # Hindi / Hinglish -> Edge-TTS Indian neural voices (Piper is English-only)
    if language in ("hi", "hinglish"):
        result = _synthesize_edge_tts(text, language)

    # English -> Local Piper TTS
    if result is None:
        result = _synthesize_piper(text, language)

    # Fallback to Edge-TTS / gTTS
    if result is None:
        result = _synthesize_edge_tts(text, language) or _synthesize_gtts(text, language)

    if result:
        elapsed = time.time() - start_time
        logger.info(f"TTS synthesis in {elapsed:.2f}s for {len(text)} chars (lang={language})")

    return result


def _synthesize_edge_tts(text: str, language: str = "en") -> Optional[Tuple[np.ndarray, int]]:
    """Synthesize multilingual speech (Hindi, Hinglish, Indian English) via edge-tts + PyAV."""
    try:
        import asyncio
        import concurrent.futures
        import av
        import edge_tts

        voice_map = {
            "hi": "hi-IN-SwaraNeural",
            "hinglish": "en-IN-NeerjaNeural",
            "en": "en-IN-NeerjaNeural",
        }
        voice = voice_map.get(language, "en-IN-NeerjaNeural")

        async def _run():
            comm = edge_tts.Communicate(text, voice)
            chunks = []
            async for msg in comm.stream():
                if msg["type"] == "audio":
                    chunks.append(msg["data"])
            return b"".join(chunks)

        # Run safely even if called inside FastAPI's running event loop
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                mp3_bytes = pool.submit(lambda: asyncio.run(_run())).result(timeout=20)
        except RuntimeError:
            mp3_bytes = asyncio.run(_run())

        if not mp3_bytes:
            return None

        container = av.open(io.BytesIO(mp3_bytes))
        resampler = av.AudioResampler(format="flt", layout="mono", rate=24000)
        pcm = []
        for frame in container.decode(audio=0):
            for rframe in resampler.resample(frame):
                pcm.append(rframe.to_ndarray().reshape(-1))
        for rframe in resampler.resample(None):
            pcm.append(rframe.to_ndarray().reshape(-1))
        container.close()

        if not pcm:
            return None
        return np.concatenate(pcm).astype(np.float32), 24000
    except Exception as e:
        logger.warning(f"Edge-TTS synthesis failed: {e}")
        return None


def _synthesize_piper(text: str, language: str) -> Optional[Tuple[np.ndarray, int]]:
    """Synthesize using Piper TTS."""
    voice = load_voice(language)
    if voice is None:
        return None
    
    try:
        chunks = []
        sr = 22050
        for chunk in voice.synthesize(text):
            if hasattr(chunk, "audio_float_array") and len(chunk.audio_float_array) > 0:
                chunks.append(chunk.audio_float_array)
                sr = getattr(chunk, "sample_rate", sr)
        if not chunks:
            return None
        audio = np.concatenate(chunks).astype(np.float32)
        return audio, sr
    except Exception as e:
        logger.error(f"Piper synthesis failed: {e}")
        return None


def _synthesize_gtts(text: str, language: str) -> Optional[Tuple[np.ndarray, int]]:
    """Fallback: synthesize using gTTS (requires internet)."""
    try:
        from gtts import gTTS
        import soundfile as sf
        
        lang_map = {"en": "en", "hi": "hi", "hinglish": "hi"}
        tts_lang = lang_map.get(language, "en")
        
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        
        tts.save(temp_path)
        
        # Convert mp3 to wav using pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_path)
            audio = audio.set_frame_rate(22050).set_channels(1)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            
            audio.export(wav_path, format="wav")
            
            audio_data, sr = _load_wav_file(wav_path)
            os.unlink(temp_path)
            os.unlink(wav_path)
            return audio_data, sr
        except Exception:
            os.unlink(temp_path)
            return None
            
    except ImportError:
        logger.warning("gTTS not available as TTS fallback")
        return None
    except Exception as e:
        logger.error(f"gTTS synthesis failed: {e}")
        return None


def _wave_file_writer(buffer):
    """Create a WAV file writer context manager."""
    import wave
    
    class WaveWriter:
        def __enter__(self):
            self.wav = wave.open(buffer, 'wb')
            self.wav.setnchannels(1)
            self.wav.setsampwidth(2)  # 16-bit
            self.wav.setframerate(22050)
            return self.wav
        
        def __exit__(self, *args):
            self.wav.close()
    
    return WaveWriter()


def _load_wav_from_buffer(buffer) -> Optional[Tuple[np.ndarray, int]]:
    """Load audio data from WAV buffer."""
    import wave
    
    buffer.seek(0)
    with wave.open(buffer, 'rb') as wav:
        sr = wav.getframerate()
        n_frames = wav.getnframes()
        raw_data = wav.readframes(n_frames)
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    return audio, sr


def _load_wav_file(path: str) -> Tuple[np.ndarray, int]:
    """Load audio from WAV file."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype='float32')
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _clean_text_for_tts(text: str, language: str) -> str:
    """Clean text for TTS synthesis."""
    import re
    
    # Remove markdown
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Limit length for TTS at sentence boundary (English '.' or Hindi '।')
    if len(text) > 500:
        cut = max(text[:500].rfind('.'), text[:500].rfind('।'))
        if cut > 50:
            text = text[:cut + 1]
        else:
            text = text[:500]
    
    return text


def audio_to_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    import wave
    
    buf = io.BytesIO()
    audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    
    return buf.getvalue()


def check_tts_health() -> dict:
    """Check TTS model availability."""
    piper_ok = _check_piper_available()
    models_dir = MODELS_DIR
    
    en_model = (models_dir / VOICE_CONFIGS["en"]["onnx"]).exists()
    
    return {
        "status": "healthy" if piper_ok else "degraded",
        "piper_available": piper_ok,
        "en_model_downloaded": en_model,
        "models_dir": str(models_dir),
        "fallback": "gTTS" if not piper_ok else None
    }

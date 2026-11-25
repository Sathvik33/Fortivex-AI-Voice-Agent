"""
Voice module: Language detection utilities.
Wraps Whisper language detection with caching and normalization.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Language code normalization map
LANGUAGE_NORM = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "hinglish": "hinglish",
    "mr": "hi",   # Marathi -> Hindi TTS fallback
    "bn": "hi",   # Bengali -> Hindi TTS fallback
    "ta": "en",   # Tamil -> English TTS fallback
    "te": "en",   # Telugu -> English TTS fallback
    "gu": "hi",   # Gujarati -> Hindi TTS fallback
    "pa": "hi",   # Punjabi -> Hindi TTS fallback
}

SUPPORTED_LANGUAGES = ["en", "hi", "hinglish"]


def normalize_language(lang_code: str) -> str:
    """Normalize language code to supported language."""
    if not lang_code:
        return "en"
    
    lang_lower = lang_code.lower().strip()
    return LANGUAGE_NORM.get(lang_lower, "en")


def detect_language_from_text(text: str) -> str:
    """
    Detect language from text using character analysis.
    Fast, no model required.
    
    Returns: "hi", "hinglish", or "en"
    """
    if not text:
        return "en"
    
    # Count Devanagari characters (Hindi script)
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len(text.replace(' ', ''))
    
    if total_chars == 0:
        return "en"
    
    devanagari_ratio = devanagari_count / total_chars
    
    if devanagari_ratio > 0.5:
        return "hi"
    elif devanagari_ratio > 0.1:
        return "hinglish"
    else:
        return "en"


def get_tts_language(detected_lang: str) -> str:
    """
    Map detected language to TTS language.
    Piper only has reliable English voice; Hindi falls back to English.
    """
    # Both hi and hinglish use English TTS (Piper limitation)
    # Document this limitation clearly
    return "en"


def get_prompt_language_instruction(lang: str) -> str:
    """Get language instruction for LLM prompt."""
    instructions = {
        "en": "Respond in English.",
        "hi": "हिंदी में जवाब दें। (Respond in Hindi)",
        "hinglish": "Hindi aur English mix karke jawab dein (Hinglish). Like: 'Aapka budget kya hai for this implementation?'",
    }
    return instructions.get(lang, "Respond in English.")

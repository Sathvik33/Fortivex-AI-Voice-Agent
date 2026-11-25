"""
Shared Groq API key pool with round-robin rotation and automatic 429 rate-limit failover.
Reads keys from:
  - GROQ_API_KEYS (comma-separated list)
  - GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, GROQ_API_KEY_4, GROQ_API_KEY_5
"""

import os
import time
import threading
import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

_lock = threading.Lock()
_rr_index = 0
_cooldown_until: dict[str, float] = {}


def get_groq_keys() -> list[str]:
    """Collect all unique, non-empty Groq API keys from environment."""
    raw: list[str] = []
    multi = os.getenv("GROQ_API_KEYS", "")
    if multi:
        raw.extend(k.strip() for k in multi.split(",") if k.strip())

    for env_name in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4", "GROQ_API_KEY_5"):
        val = os.getenv(env_name, "").strip()
        if val:
            raw.append(val)

    seen = set()
    unique = []
    for k in raw:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def has_groq_keys() -> bool:
    """Return True if at least one Groq API key is configured."""
    return len(get_groq_keys()) > 0


def _ordered_keys() -> list[str]:
    """Return keys rotated by round-robin index, prioritizing keys not in cooldown."""
    global _rr_index
    keys = get_groq_keys()
    if not keys:
        return []

    now = time.time()
    with _lock:
        start = _rr_index % len(keys)
        _rr_index = (start + 1) % len(keys)

    rotated = keys[start:] + keys[:start]
    ready = [k for k in rotated if _cooldown_until.get(k, 0) <= now]
    cooling = [k for k in rotated if _cooldown_until.get(k, 0) > now]
    return ready + cooling


def mark_rate_limited(api_key: str, seconds: float = 60.0):
    """Put a rate-limited key into temporary cooldown."""
    with _lock:
        _cooldown_until[api_key] = time.time() + seconds
    logger.warning(f"Groq key ...{api_key[-6:]} rate-limited; cooling down for {seconds:.0f}s")


def call_with_groq_key(fn: Callable[[str], T]) -> T:
    """
    Execute `fn(api_key)` across available Groq keys, failing over on 429/rate-limit errors.
    `fn` should raise an exception (e.g. requests.HTTPError) on failure.
    """
    keys = _ordered_keys()
    if not keys:
        raise RuntimeError("No Groq API keys configured")

    last_err = None
    for idx, key in enumerate(keys):
        try:
            return fn(key)
        except Exception as e:
            last_err = e
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            err_str = str(e).lower()
            if status_code == 429 or "rate limit" in err_str or "429" in err_str:
                mark_rate_limited(key, 60.0)
                if idx < len(keys) - 1:
                    continue
            elif status_code in (401, 403) and idx < len(keys) - 1:
                mark_rate_limited(key, 300.0)
                continue
            raise
    raise last_err

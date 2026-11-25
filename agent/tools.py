"""
LangGraph agent tools.
Functions the graph nodes call to interact with external systems.
"""

import json
import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if GROQ_API_KEY else "ollama")


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by Qwen3 models."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _clean_spoken_response(text: str) -> str:
    """Strip meta-labels like '(For voice response: ...)' and duplicate paragraphs."""
    text = _strip_think_tags(text)
    # Remove trailing parenthetical or labelled voice-response duplicates
    text = re.sub(r"\(\s*For voice response\s*:[\s\S]*?\)\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"(?:\n+|\s+)For voice response\s*:[\s\S]*$", "", text, flags=re.IGNORECASE).strip()

    # Deduplicate identical paragraphs if the model still repeats itself
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    deduped = []
    for p in paras:
        if p not in deduped:
            deduped.append(p)
    return "\n\n".join(deduped).strip()


def _groq_generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 800,
    timeout: int = 30
) -> str:
    """Call Groq OpenAI-compatible chat completions API with multi-key rotation."""
    import requests
    from services.groq_keys import call_with_groq_key

    model = os.getenv("GROQ_MODEL", GROQ_MODEL)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    def _do_call(api_key: str) -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max(max_tokens, 600),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return _strip_think_tags(data["choices"][0]["message"]["content"])

    return call_with_groq_key(_do_call)


def ollama_generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 500,
    timeout: int = 60
) -> str:
    """Generate LLM response using Groq (when configured) with Ollama fallback."""
    import requests
    from services.groq_keys import has_groq_keys

    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()

    if provider == "groq" and has_groq_keys():
        try:
            return _groq_generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens, timeout=min(timeout, 30))
        except Exception as e:
            logger.warning(f"Groq generation failed ({e}), falling back to Ollama")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return _strip_think_tags(data.get("message", {}).get("content", ""))
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        raise


def parse_json_response(text: str, default: dict = None) -> dict:
    """Safely parse JSON from LLM output."""
    if default is None:
        default = {}
    
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON block
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if pattern.startswith('`') else match.group(0))
            except json.JSONDecodeError:
                continue
    
    logger.warning(f"Could not parse JSON from: {text[:100]}")
    return default


def classify_intent(message: str, history: list[dict]) -> dict:
    """Classify customer intent using LLM."""
    from agent.prompts import INTENT_CLASSIFICATION_PROMPT
    
    # Format recent context
    context = "\n".join([
        f"{m['role'].title()}: {m['content'][:100]}"
        for m in history[-4:]
    ])
    
    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        message=message,
        context=context or "No previous context"
    )
    
    try:
        response = ollama_generate(prompt, temperature=0.1, max_tokens=200, timeout=30)
        result = parse_json_response(response, {
            "intent": "general",
            "confidence": 0.5,
            "key_topics": []
        })
        return result
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}")
        return {"intent": "general", "confidence": 0.5, "key_topics": []}


def extract_entities(message: str, existing_data: dict) -> dict:
    """Extract customer information entities from message using LLM."""
    from agent.prompts import ENTITY_EXTRACTION_PROMPT
    
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        message=message,
        existing_data=json.dumps(existing_data, indent=2)
    )
    
    try:
        response = ollama_generate(prompt, temperature=0.0, max_tokens=400, timeout=30)
        return parse_json_response(response, {})
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return {}


def get_llm_sentiment(message: str, previous: dict) -> dict:
    """Get LLM-based sentiment analysis."""
    from agent.prompts import SENTIMENT_ANALYSIS_PROMPT
    
    prompt = SENTIMENT_ANALYSIS_PROMPT.format(
        message=message,
        previous_sentiment=json.dumps(previous)
    )
    
    try:
        response = ollama_generate(prompt, temperature=0.1, max_tokens=300, timeout=30)
        return parse_json_response(response, {
            "overall": "neutral",
            "buying_intent": "low",
            "frustration_level": "none",
            "interest_level": "medium",
            "urgency": "low",
            "objections": [],
            "buying_signals": []
        })
    except Exception as e:
        logger.warning(f"LLM sentiment failed: {e}")
        return {}


def generate_rag_response(
    question: str,
    context: str,
    customer_name: str,
    language: str
) -> str:
    """Generate RAG-grounded response."""
    from agent.prompts import RAG_RESPONSE_PROMPT
    
    prompt = RAG_RESPONSE_PROMPT.format(
        context=context,
        question=question,
        customer_name=customer_name or "there",
        language=language
    )
    
    try:
        return _clean_spoken_response(ollama_generate(prompt, temperature=0.5, max_tokens=400, timeout=60))
    except Exception as e:
        logger.error(f"RAG response generation failed: {e}")
        return "I apologize, I'm having trouble accessing our knowledge base right now. Please try again."


def generate_conversational_response(
    message: str,
    history: list[dict],
    customer_context: dict,
    intent: str,
    language: str,
    rag_context: str = ""
) -> str:
    """Generate a conversational sales response."""
    from agent.prompts import GENERAL_RESPONSE_PROMPT, SYSTEM_PERSONA
    from voice.language import get_prompt_language_instruction
    
    # Format history
    history_text = "\n".join([
        f"{m['role'].title()}: {m['content']}"
        for m in history[-8:]
    ])
    
    # RAG context injection
    rag_section = ""
    if rag_context:
        rag_section = f"\nRelevant knowledge base context:\n{rag_context}\n\nUse this context to answer accurately."
    
    lang_instruction = get_prompt_language_instruction(language)
    
    prompt = GENERAL_RESPONSE_PROMPT.format(
        system_prompt=SYSTEM_PERSONA,
        history=history_text or "No previous conversation",
        customer_context=json.dumps({
            k: v for k, v in customer_context.items() if v is not None
        }, indent=2),
        message=message,
        intent=intent,
        language=f"{language} - {lang_instruction}",
        rag_context=rag_section
    )
    
    try:
        return _clean_spoken_response(ollama_generate(prompt, temperature=0.7, max_tokens=500, timeout=60))
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "I apologize, I'm experiencing a technical issue. Could you please repeat that?"


def check_ollama_health() -> dict:
    """Check active LLM provider health (Groq or local Ollama)."""
    import requests

    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if provider == "groq" and api_key:
        return {
            "status": "healthy",
            "provider": "groq",
            "model": os.getenv("GROQ_MODEL", GROQ_MODEL),
        }
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            model_available = any(OLLAMA_MODEL in name for name in model_names)
            
            return {
                "status": "healthy" if model_available else "model_missing",
                "ollama_running": True,
                "model": OLLAMA_MODEL,
                "model_available": model_available,
                "available_models": model_names
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_running": False,
            "error": str(e)
        }

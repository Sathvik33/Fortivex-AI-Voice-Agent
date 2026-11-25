"""
Sentiment analysis engine for customer conversations.
Uses LLM for extraction + deterministic rule-based classification.
"""

import json
import logging
import re
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Complete sentiment analysis result."""
    overall: str = "neutral"           # positive, neutral, negative
    buying_intent: str = "low"         # high, medium, low
    frustration_level: str = "none"    # high, medium, low, none
    objections: list = field(default_factory=list)
    interest_level: str = "medium"     # high, medium, low
    urgency: str = "low"               # high, medium, low
    buying_signals: list = field(default_factory=list)
    confidence: float = 0.8


# Keyword dictionaries for rule-based detection
POSITIVE_SIGNALS = [
    "interested", "love", "great", "excellent", "perfect", "exactly what",
    "sounds good", "that works", "yes", "definitely", "absolutely",
    "looking forward", "excited", "impressive", "wonderful", "amazing",
    "good", "nice", "helpful", "useful", "want", "need this",
    "हाँ", "बढ़िया", "अच्छा", "चाहिए", "ज़रूर"
]

NEGATIVE_SIGNALS = [
    "not interested", "too expensive", "can't afford", "no thanks",
    "disappointed", "frustrated", "bad", "terrible", "awful",
    "waste", "useless", "doesn't work", "problem", "issue", "annoying",
    "नहीं", "महंगा", "समस्या", "परेशान"
]

BUYING_INTENT_HIGH = [
    "ready to buy", "want to purchase", "let's proceed", "how do i buy",
    "can we start", "ready to sign", "book a demo", "schedule", "demo",
    "pricing", "cost", "quote", "proposal", "implementation",
    "कब शुरू", "खरीदना है", "demo चाहिए"
]

FRUSTRATION_HIGH = [
    "speak to human", "real person", "supervisor", "manager",
    "this is useless", "not helpful", "wasting my time", "already told you",
    "how many times", "you don't understand", "stupid bot",
    "इंसान से बात", "मैनेजर", "बेकार"
]

OBJECTION_PATTERNS = [
    r"too expensive|not in budget|can't afford|price is high",
    r"need to think|not sure|not ready|maybe later",
    r"already have|using different|happy with current",
    r"need approval|talk to.*team|consult.*colleague",
    r"features? (missing|lack|don't have|not enough)",
    r"महंगा|बजट नहीं|सोचना है|पहले से है"
]

URGENCY_HIGH = [
    "asap", "immediately", "urgent", "right away", "this week",
    "today", "right now", "as soon as possible", "can't wait",
    "तुरंत", "अभी", "जल्दी", "फौरन"
]


def analyze_sentiment_rules(message: str) -> SentimentResult:
    """
    Rule-based sentiment analysis as a fallback or complement to LLM.
    Fast, deterministic, no model required.
    """
    result = SentimentResult()
    message_lower = message.lower()
    
    # Overall sentiment
    pos_count = sum(1 for sig in POSITIVE_SIGNALS if sig.lower() in message_lower)
    neg_count = sum(1 for sig in NEGATIVE_SIGNALS if sig.lower() in message_lower)
    
    if pos_count > neg_count:
        result.overall = "positive"
    elif neg_count > pos_count:
        result.overall = "negative"
    else:
        result.overall = "neutral"
    
    # Buying intent
    intent_signals = sum(1 for sig in BUYING_INTENT_HIGH if sig.lower() in message_lower)
    if intent_signals >= 2:
        result.buying_intent = "high"
    elif intent_signals == 1:
        result.buying_intent = "medium"
    else:
        result.buying_intent = "low"
    
    # Frustration
    frustration_signals = sum(1 for sig in FRUSTRATION_HIGH if sig.lower() in message_lower)
    if frustration_signals >= 2:
        result.frustration_level = "high"
    elif frustration_signals == 1:
        result.frustration_level = "medium"
    elif neg_count >= 2:
        result.frustration_level = "low"
    else:
        result.frustration_level = "none"
    
    # Objections
    result.objections = []
    for pattern in OBJECTION_PATTERNS:
        if re.search(pattern, message_lower):
            match = re.search(pattern, message_lower)
            if match:
                result.objections.append(match.group(0))
    
    # Urgency
    urgency_signals = sum(1 for sig in URGENCY_HIGH if sig.lower() in message_lower)
    if urgency_signals >= 1:
        result.urgency = "high"
    
    # Interest level
    if pos_count >= 2 or intent_signals >= 1:
        result.interest_level = "high"
    elif neg_count >= 2:
        result.interest_level = "low"
    else:
        result.interest_level = "medium"
    
    # Buying signals
    result.buying_signals = [sig for sig in BUYING_INTENT_HIGH if sig.lower() in message_lower]
    
    return result


def parse_llm_sentiment(llm_output: str) -> Optional[SentimentResult]:
    """Parse LLM-generated sentiment JSON into SentimentResult."""
    try:
        # Extract JSON from LLM output
        json_match = re.search(r'\{[^{}]*\}', llm_output, re.DOTALL)
        if not json_match:
            return None
        
        data = json.loads(json_match.group(0))
        
        result = SentimentResult(
            overall=data.get("overall", "neutral"),
            buying_intent=data.get("buying_intent", "low"),
            frustration_level=data.get("frustration_level", "none"),
            objections=data.get("objections", []),
            interest_level=data.get("interest_level", "medium"),
            urgency=data.get("urgency", "low"),
            buying_signals=data.get("buying_signals", []),
        )
        return result
    except Exception as e:
        logger.warning(f"Could not parse LLM sentiment: {e}")
        return None


def merge_sentiment_results(rule_based: SentimentResult, llm_based: Optional[SentimentResult]) -> SentimentResult:
    """
    Merge rule-based and LLM sentiment results.
    LLM takes priority for overall/buying_intent; rules take priority for frustration.
    """
    if llm_based is None:
        return rule_based
    
    merged = SentimentResult(
        overall=llm_based.overall,
        buying_intent=llm_based.buying_intent,
        # Frustration: use whichever detects higher level
        frustration_level=_higher_level(rule_based.frustration_level, llm_based.frustration_level, 
                                         ["none", "low", "medium", "high"]),
        objections=list(set(rule_based.objections + llm_based.objections)),
        interest_level=llm_based.interest_level,
        urgency=_higher_level(rule_based.urgency, llm_based.urgency, ["low", "medium", "high"]),
        buying_signals=list(set(rule_based.buying_signals + llm_based.buying_signals)),
    )
    return merged


def _higher_level(a: str, b: str, scale: list) -> str:
    """Return whichever value is higher on the scale."""
    idx_a = scale.index(a) if a in scale else 0
    idx_b = scale.index(b) if b in scale else 0
    return scale[max(idx_a, idx_b)]


def sentiment_to_dict(result: SentimentResult) -> dict:
    """Convert SentimentResult to dict for storage."""
    return {
        "overall": result.overall,
        "buying_intent": result.buying_intent,
        "frustration_level": result.frustration_level,
        "objections": result.objections,
        "interest_level": result.interest_level,
        "urgency": result.urgency,
        "buying_signals": result.buying_signals,
    }


def should_escalate_to_human(sentiment: SentimentResult) -> bool:
    """Determine if the conversation should be escalated to a human agent."""
    return (
        sentiment.frustration_level in ["high", "medium"] and
        sentiment.overall == "negative"
    )

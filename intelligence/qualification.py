"""
Intelligence module: Lead Qualification Engine.
Deterministic scoring based on extracted customer information.
No LLM-invented scores — Python calculates all scores.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QualificationResult:
    """Complete lead qualification result."""
    # Individual scores
    budget_score: float = 0.0        # Max 25
    timeline_score: float = 0.0      # Max 20
    authority_score: float = 0.0     # Max 20
    requirement_score: float = 0.0   # Max 20
    engagement_score: float = 0.0    # Max 15
    
    # Totals
    total_score: float = 0.0
    lead_category: str = "COLD"
    
    # Explanation
    score_breakdown: dict = field(default_factory=dict)
    score_explanation: str = ""
    missing_information: list = field(default_factory=list)


def calculate_lead_score(customer_context: dict, qualification_signals: dict = None) -> QualificationResult:
    """
    Calculate deterministic lead score from 0-100.
    
    Scoring breakdown:
    - Budget availability: 25 points
    - Purchase timeline: 20 points
    - Decision-making authority: 20 points
    - Product requirement match: 20 points
    - Engagement and buying intent: 15 points
    """
    result = QualificationResult()
    breakdown = {}
    missing = []
    
    # ─── BUDGET SCORE (0–25) ─────────────────────────────────────────────────
    budget = customer_context.get("budget")
    budget_currency = customer_context.get("budget_currency", "INR")
    
    if budget is not None and budget > 0:
        # Normalize to INR for comparison
        budget_inr = _to_inr(float(budget), budget_currency)
        
        if budget_inr >= 1_000_000:        # >= 10 lakh
            result.budget_score = 25.0
            breakdown["budget"] = "High budget (≥10L INR): 25/25"
        elif budget_inr >= 500_000:        # >= 5 lakh
            result.budget_score = 20.0
            breakdown["budget"] = "Good budget (≥5L INR): 20/25"
        elif budget_inr >= 200_000:        # >= 2 lakh
            result.budget_score = 15.0
            breakdown["budget"] = "Adequate budget (≥2L INR): 15/25"
        elif budget_inr >= 100_000:        # >= 1 lakh
            result.budget_score = 10.0
            breakdown["budget"] = "Limited budget (≥1L INR): 10/25"
        else:
            result.budget_score = 5.0
            breakdown["budget"] = "Low budget (<1L INR): 5/25"
    else:
        result.budget_score = 0.0
        breakdown["budget"] = "Budget not provided: 0/25"
        missing.append("budget")
    
    # ─── TIMELINE SCORE (0–20) ───────────────────────────────────────────────
    timeline = customer_context.get("purchase_timeline", "")
    if timeline:
        timeline_lower = str(timeline).lower()
        
        if any(word in timeline_lower for word in ["immediately", "asap", "urgent", "this month", "this week"]):
            result.timeline_score = 20.0
            breakdown["timeline"] = "Immediate/urgent timeline: 20/20"
        elif any(word in timeline_lower for word in ["1 month", "2 month", "one month", "two month", "60 day", "30 day"]):
            result.timeline_score = 18.0
            breakdown["timeline"] = "1-2 month timeline: 18/20"
        elif any(word in timeline_lower for word in ["3 month", "quarter", "q1", "q2", "q3", "q4", "three month"]):
            result.timeline_score = 14.0
            breakdown["timeline"] = "3 month / quarterly timeline: 14/20"
        elif any(word in timeline_lower for word in ["6 month", "half year", "six month"]):
            result.timeline_score = 10.0
            breakdown["timeline"] = "6 month timeline: 10/20"
        elif any(word in timeline_lower for word in ["year", "annual", "12 month"]):
            result.timeline_score = 5.0
            breakdown["timeline"] = "Annual/long-term timeline: 5/20"
        else:
            result.timeline_score = 8.0
            breakdown["timeline"] = "Timeline mentioned but unclear: 8/20"
    else:
        result.timeline_score = 0.0
        breakdown["timeline"] = "Timeline not provided: 0/20"
        missing.append("purchase_timeline")
    
    # ─── AUTHORITY SCORE (0–20) ──────────────────────────────────────────────
    is_decision_maker = customer_context.get("is_decision_maker")
    
    if is_decision_maker is True:
        result.authority_score = 20.0
        breakdown["authority"] = "Confirmed decision maker: 20/20"
    elif is_decision_maker is False:
        result.authority_score = 8.0
        breakdown["authority"] = "Not the decision maker: 8/20"
    else:
        # Check for titles in name or company context
        name = str(customer_context.get("name", "")).lower()
        requirements = str(customer_context.get("requirements", "")).lower()
        
        if any(title in name + requirements for title in ["ceo", "cto", "coo", "director", "vp", "head", "owner", "founder", "president"]):
            result.authority_score = 18.0
            breakdown["authority"] = "Decision maker title detected: 18/20"
        else:
            result.authority_score = 10.0
            breakdown["authority"] = "Decision-making authority unknown: 10/20"
            missing.append("decision_making_authority")
    
    # ─── REQUIREMENT SCORE (0–20) ────────────────────────────────────────────
    requirements = customer_context.get("requirements", "")
    company = customer_context.get("company", "")
    company_size = customer_context.get("company_size")
    product_interests = customer_context.get("product_interests", "")
    
    req_score = 0
    if requirements and len(str(requirements)) > 20:
        req_score += 10
    elif requirements:
        req_score += 5
    else:
        missing.append("requirements")
    
    if company:
        req_score += 3
    if company_size and int(company_size) > 0:
        req_score += 3
        if int(company_size) >= 100:
            req_score += 2  # Enterprise customer
    if product_interests:
        req_score += 2
    
    result.requirement_score = min(20.0, float(req_score))
    breakdown["requirements"] = f"Requirements clarity: {result.requirement_score}/20"
    
    # ─── ENGAGEMENT SCORE (0–15) ─────────────────────────────────────────────
    # Based on qualification signals from LLM analysis
    eng_score = 7.0  # Base score for being in conversation
    
    if qualification_signals:
        if qualification_signals.get("buying_signals"):
            eng_score += min(4.0, len(qualification_signals["buying_signals"]) * 2.0)
        if qualification_signals.get("engagement_high"):
            eng_score += 2.0
        if qualification_signals.get("product_match"):
            eng_score += 2.0
    
    result.engagement_score = min(15.0, eng_score)
    breakdown["engagement"] = f"Engagement level: {result.engagement_score}/15"
    
    # ─── TOTAL & CATEGORY ────────────────────────────────────────────────────
    result.total_score = round(
        result.budget_score + result.timeline_score +
        result.authority_score + result.requirement_score +
        result.engagement_score, 2
    )
    
    if result.total_score >= 80:
        result.lead_category = "HOT"
    elif result.total_score >= 50:
        result.lead_category = "WARM"
    else:
        result.lead_category = "COLD"
    
    result.score_breakdown = breakdown
    result.missing_information = missing
    
    result.score_explanation = _build_explanation(result, breakdown, missing)
    
    return result


def _to_inr(amount: float, currency: str) -> float:
    """Convert budget to INR for standardized comparison."""
    rates = {
        "INR": 1.0,
        "USD": 83.0,
        "EUR": 90.0,
        "GBP": 105.0,
        "AED": 22.6,
        "SGD": 62.0,
    }
    return amount * rates.get(currency.upper(), 1.0)


def _build_explanation(result: QualificationResult, breakdown: dict, missing: list) -> str:
    """Build a human-readable explanation of the score."""
    lines = [
        f"Lead Score: {result.total_score}/100 ({result.lead_category})",
        "",
        "Score Breakdown:",
    ]
    for key, desc in breakdown.items():
        lines.append(f"  • {desc}")
    
    if missing:
        lines.append("")
        lines.append(f"Missing Information: {', '.join(missing)}")
        lines.append("Collecting these details would improve the score.")
    
    return "\n".join(lines)


def get_next_qualification_question(customer_context: dict) -> Optional[str]:
    """
    Determine what information to gather next.
    Returns a natural question to ask, or None if fully qualified.
    """
    missing_priority = [
        ("name", "Could I get your name please?"),
        ("company", "What company are you from?"),
        ("requirements", "What specific features or capabilities are you looking for in a CRM solution?"),
        ("budget", "Do you have a budget range in mind for this implementation?"),
        ("purchase_timeline", "What's your target timeline for getting this implemented?"),
        ("company_size", "How many employees does your company have?"),
        ("is_decision_maker", "Are you the primary decision maker for this purchase, or will others be involved?"),
    ]
    
    for field, question in missing_priority:
        value = customer_context.get(field)
        if value is None or value == "" or value == 0:
            return question
    
    return None  # Fully qualified

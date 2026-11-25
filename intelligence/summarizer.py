"""
Intelligence module: Call summarizer using local Ollama LLM.
Generates structured summaries at the end of each call.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def generate_call_summary(
    conversation_history: list[dict],
    customer_context: dict,
    lead_score: float,
    lead_category: str,
    sentiment: dict,
    appointment: Optional[dict] = None,
    llm_fn=None
) -> Optional[dict]:
    """
    Generate a structured call summary using the LLM.
    
    Args:
        conversation_history: List of {role, content} messages
        customer_context: Extracted customer information
        lead_score: Calculated lead score (0-100)
        lead_category: HOT/WARM/COLD
        sentiment: Sentiment analysis result
        appointment: Booked appointment dict or None
        llm_fn: Function to call LLM (ollama_generate)
    
    Returns:
        Structured summary dict
    """
    from agent.prompts import CALL_SUMMARY_PROMPT
    
    # Format transcript
    transcript_lines = []
    for msg in conversation_history[-30:]:  # Last 30 messages
        role = "Customer" if msg.get("role") == "user" else "Agent"
        content = msg.get("content", "")
        if content:
            transcript_lines.append(f"{role}: {content}")
    
    transcript = "\n".join(transcript_lines)
    
    if not transcript:
        return _default_summary(customer_context, lead_score, lead_category, sentiment, appointment)
    
    # Format appointment info
    appt_info = "No appointment booked"
    if appointment:
        appt_info = f"Appointment confirmed for {appointment.get('datetime_display', 'unknown time')}"
    
    # Format sentiment
    sentiment_str = f"{sentiment.get('overall', 'neutral')} (buying intent: {sentiment.get('buying_intent', 'low')})"
    
    prompt = CALL_SUMMARY_PROMPT.format(
        transcript=transcript,
        customer_context=json.dumps(customer_context, indent=2),
        lead_score=lead_score,
        lead_category=lead_category,
        sentiment=sentiment_str,
        appointment_info=appt_info
    )
    
    if llm_fn is None:
        return _default_summary(customer_context, lead_score, lead_category, sentiment, appointment)
    
    try:
        response = llm_fn(prompt, temperature=0.3, max_tokens=1000)
        return _parse_summary_json(response, customer_context, lead_score, lead_category, sentiment, appointment)
    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        return _default_summary(customer_context, lead_score, lead_category, sentiment, appointment)


def _parse_summary_json(
    llm_output: str,
    customer_context: dict,
    lead_score: float,
    lead_category: str,
    sentiment: dict,
    appointment: Optional[dict]
) -> dict:
    """Parse LLM summary output into structured dict."""
    # Extract JSON from LLM response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_output, re.DOTALL)
    
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            # Ensure required fields
            parsed["lead_score"] = lead_score
            parsed["lead_category"] = lead_category
            if appointment:
                parsed["appointment_booked"] = True
                parsed["appointment_id"] = appointment.get("id")
            return parsed
        except json.JSONDecodeError:
            pass
    
    # Fallback to default if parsing fails
    return _default_summary(customer_context, lead_score, lead_category, sentiment, appointment)


def _default_summary(
    customer_context: dict,
    lead_score: float,
    lead_category: str,
    sentiment: dict,
    appointment: Optional[dict]
) -> dict:
    """Generate a basic summary from available data without LLM."""
    name = customer_context.get("name", "Unknown Customer")
    company = customer_context.get("company", "Unknown Company")
    budget = customer_context.get("budget")
    budget_str = f"₹{budget:,.0f}" if budget else "Not specified"
    
    # Determine recommended action
    if lead_category == "HOT":
        action = f"Priority follow-up with {name}. Close deal immediately. Send formal proposal."
    elif lead_category == "WARM":
        action = f"Schedule follow-up with {name} within 2-3 days. Send product brochure and proposal."
    else:
        action = f"Nurture {name} with product information. Follow up in 1-2 weeks."
    
    if appointment:
        action = f"Appointment confirmed. Prepare demo. {action}"
    
    return {
        "customer_name": name,
        "company": company,
        "requirements": customer_context.get("requirements", "Not specified"),
        "budget": budget_str,
        "purchase_timeline": customer_context.get("purchase_timeline", "Not specified"),
        "product_interests": customer_context.get("product_interests", "CRM/ERP solutions"),
        "objections": customer_context.get("objections", "None mentioned"),
        "sentiment_summary": f"Overall {sentiment.get('overall', 'neutral')}, buying intent {sentiment.get('buying_intent', 'low')}",
        "lead_score": lead_score,
        "lead_category": lead_category,
        "appointment_details": appointment.get("datetime_display") if appointment else "None",
        "appointment_booked": appointment is not None,
        "appointment_id": appointment.get("id") if appointment else None,
        "recommended_action": action,
        "key_insights": [
            f"Lead category: {lead_category} ({lead_score}/100)",
            f"Budget: {budget_str}",
            f"Timeline: {customer_context.get('purchase_timeline', 'Not specified')}",
        ],
        "full_summary": (
            f"{name} from {company} contacted TechVex Solutions regarding enterprise software. "
            f"Budget: {budget_str}. Timeline: {customer_context.get('purchase_timeline', 'Not specified')}. "
            f"Lead score: {lead_score}/100 ({lead_category}). "
            f"Sentiment: {sentiment.get('overall', 'neutral')}. "
            f"{'Appointment booked.' if appointment else 'No appointment booked.'}"
        )
    }


def save_call_summary(
    conversation_id: str,
    customer_id: Optional[str],
    summary: dict
) -> bool:
    """Save call summary to database."""
    from database.db import execute_query
    import psycopg2.extras
    
    try:
        execute_query(
            """
            INSERT INTO call_summaries (
                conversation_id, customer_id, customer_name, company,
                requirements, budget, purchase_timeline, product_interests,
                objections, sentiment, lead_score, lead_category,
                appointment_booked, appointment_id, recommended_action, full_summary
            ) VALUES (
                %s::uuid, %s::uuid, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            """,
            (
                conversation_id,
                customer_id,
                summary.get("customer_name"),
                summary.get("company"),
                summary.get("requirements"),
                summary.get("budget"),
                summary.get("purchase_timeline"),
                summary.get("product_interests"),
                summary.get("objections"),
                summary.get("sentiment_summary"),
                summary.get("lead_score", 0),
                summary.get("lead_category", "COLD"),
                summary.get("appointment_booked", False),
                summary.get("appointment_id"),
                summary.get("recommended_action"),
                summary.get("full_summary")
            )
        )
        logger.info(f"Saved call summary for conversation {conversation_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save call summary: {e}")
        return False

"""
LangGraph Agent Graph: Autonomous Sales Voice Agent.
Implements the full agent workflow with conditional routing.
"""

import json
import logging
import uuid
from typing import Any, Optional
from datetime import datetime, date, time as time_type

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.tools import (
    classify_intent, extract_entities, get_llm_sentiment,
    generate_rag_response, generate_conversational_response,
    ollama_generate
)
from intelligence.qualification import calculate_lead_score
from intelligence.sentiment import (
    analyze_sentiment_rules, merge_sentiment_results,
    parse_llm_sentiment, sentiment_to_dict, should_escalate_to_human
)
from intelligence.memory import SessionStore
from voice.language import detect_language_from_text, normalize_language

logger = logging.getLogger(__name__)


# ─── NODE FUNCTIONS ───────────────────────────────────────────────────────────

def node_load_customer_memory(state: AgentState) -> AgentState:
    """Load or initialize customer memory from session store."""
    session_id = state.get("session_id", str(uuid.uuid4()))
    customer_id = state.get("customer_id")
    
    memory = SessionStore.get_or_create(session_id, customer_id)
    
    return {
        **state,
        "session_id": session_id,
        "customer_context": memory.context,
        "detected_language": memory.detected_language,
    }


def node_detect_language(state: AgentState) -> AgentState:
    """Detect language from latest message."""
    messages = state.get("messages", [])
    if not messages:
        return state
    
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    detected = detect_language_from_text(content)
    
    # Update session memory
    memory = SessionStore.get(state["session_id"])
    if memory:
        memory.detected_language = detected
    
    return {**state, "detected_language": detected}


def node_classify_intent(state: AgentState) -> AgentState:
    """Classify customer intent from latest message."""
    messages = state.get("messages", [])
    if not messages:
        return {**state, "detected_intent": "general"}
    
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    history = state.get("conversation_history", [])
    
    try:
        intent_result = classify_intent(content, history)
        intent = intent_result.get("intent", "general")
    except Exception as e:
        logger.warning(f"Intent classification error: {e}")
        intent = "general"
    
    # Determine routing flags
    should_retrieve = intent in ["product_inquiry", "pricing_inquiry", "faq"]
    should_qualify = intent in ["information_sharing", "greeting", "general"]
    should_book = intent in ["appointment_request", "appointment_confirmation"]
    should_escalate = intent in ["escalation_request", "complaint"]
    
    return {
        **state,
        "detected_intent": intent,
        "should_retrieve_rag": should_retrieve,
        "should_qualify": should_qualify,
        "should_book": should_book,
        "should_escalate": should_escalate,
        "next_action": intent
    }


def node_retrieve_knowledge(state: AgentState) -> AgentState:
    """Retrieve relevant documents from knowledge base."""
    if not state.get("should_retrieve_rag"):
        return {**state, "retrieved_documents": [], "rag_context": ""}
    
    messages = state.get("messages", [])
    if not messages:
        return state
    
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    try:
        from rag.retriever import retrieve_relevant_chunks, format_rag_context
        
        chunks = retrieve_relevant_chunks(query, top_k=5)
        rag_context = format_rag_context(chunks)
        
        return {
            **state,
            "retrieved_documents": chunks,
            "rag_context": rag_context
        }
    except Exception as e:
        logger.warning(f"RAG retrieval error: {e}")
        return {**state, "retrieved_documents": [], "rag_context": ""}


def node_extract_customer_information(state: AgentState) -> AgentState:
    """Extract customer information from latest message."""
    messages = state.get("messages", [])
    if not messages:
        return state
    
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    customer_context = state.get("customer_context", {}) or {}
    
    try:
        extracted = extract_entities(content, customer_context)
        
        if extracted:
            # Update session memory
            memory = SessionStore.get(state["session_id"])
            if memory:
                memory.update_context(extracted)
                customer_context = memory.context.copy()
            else:
                customer_context.update({k: v for k, v in extracted.items() if v is not None})
            
            logger.info(f"Extracted entities: {list(extracted.keys())}")
    except Exception as e:
        logger.warning(f"Entity extraction error: {e}")
    
    return {**state, "customer_context": customer_context}


def node_qualify_lead(state: AgentState) -> AgentState:
    """Calculate lead score from customer context."""
    customer_context = state.get("customer_context", {}) or {}
    
    try:
        result = calculate_lead_score(customer_context)
        
        qual_data = {
            "budget_score": result.budget_score,
            "timeline_score": result.timeline_score,
            "authority_score": result.authority_score,
            "requirement_score": result.requirement_score,
            "engagement_score": result.engagement_score,
            "total_score": result.total_score,
            "lead_category": result.lead_category,
            "score_explanation": result.score_explanation,
            "missing_information": result.missing_information
        }
        
        return {
            **state,
            "qualification_data": qual_data,
            "lead_score": result.total_score
        }
    except Exception as e:
        logger.warning(f"Lead qualification error: {e}")
        return {**state, "lead_score": 0.0}


def node_analyze_sentiment(state: AgentState) -> AgentState:
    """Analyze sentiment from latest message."""
    messages = state.get("messages", [])
    if not messages:
        return state
    
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    previous_sentiment = state.get("sentiment", {}) or {}
    
    # Rule-based (always runs)
    rule_sentiment = analyze_sentiment_rules(content)
    
    # LLM-based (best effort)
    llm_sentiment_dict = {}
    try:
        llm_sentiment_dict = get_llm_sentiment(content, previous_sentiment)
        llm_sentiment = parse_llm_sentiment(json.dumps(llm_sentiment_dict))
    except Exception:
        llm_sentiment = None
    
    # Merge results
    merged = merge_sentiment_results(rule_sentiment, llm_sentiment)
    sentiment_dict = sentiment_to_dict(merged)
    
    # Update memory
    memory = SessionStore.get(state["session_id"])
    if memory:
        memory.add_sentiment(sentiment_dict)
    
    # Check escalation
    if should_escalate_to_human(merged):
        return {
            **state,
            "sentiment": sentiment_dict,
            "should_escalate": True
        }
    
    return {**state, "sentiment": sentiment_dict}


def node_check_appointment_availability(state: AgentState) -> AgentState:
    """Check available appointment slots."""
    if not state.get("should_book"):
        return {**state, "available_slots": []}
    
    try:
        from services.calendar import get_available_slots
        slots = get_available_slots(days_ahead=10, limit=6)
        
        return {**state, "available_slots": slots}
    except Exception as e:
        logger.warning(f"Slot check error: {e}")
        return {**state, "available_slots": []}


def node_confirm_appointment(state: AgentState) -> AgentState:
    """Handle appointment confirmation from customer."""
    intent = state.get("detected_intent", "")
    messages = state.get("messages", [])
    if not messages:
        return state

    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    history = state.get("conversation_history", []) or []

    from services.calendar import parse_slot_preference, has_explicit_slot_choice

    # Check if agent previously offered slots in this conversation
    slots_offered_previously = any(
        m.get("role") == "assistant" and "available slots" in m.get("content", "").lower()
        for m in history[-4:]
    )

    is_confirmation = (
        intent == "appointment_confirmation"
        or (intent == "appointment_request" and slots_offered_previously and has_explicit_slot_choice(content))
    )

    if not is_confirmation:
        return state

    available_slots = state.get("available_slots", [])
    booking_request = state.get("booking_request", {}) or {}

    if available_slots:
        matched_slot = parse_slot_preference(content, available_slots)
        if matched_slot:
            booking_request["confirmed"] = True
            booking_request["slot"] = matched_slot

    return {**state, "booking_request": booking_request}


def node_create_appointment(state: AgentState) -> AgentState:
    """Create a confirmed appointment."""
    booking_request = state.get("booking_request", {}) or {}

    if not booking_request.get("confirmed") or not booking_request.get("slot"):
        return state

    customer_id = state.get("customer_id")
    if not customer_id:
        # Ensure a customer record exists in CRM before creating the appointment
        try:
            from services.crm import create_or_update_customer
            ctx = state.get("customer_context", {}) or {}
            crm_payload = {
                **ctx,
                "name": ctx.get("name") or "Voice Prospect",
                "preferred_language": state.get("detected_language", "en"),
            }
            customer_id = create_or_update_customer(crm_payload)
        except Exception as e:
            logger.error(f"Could not create customer for appointment: {e}")
            return state

    if not customer_id:
        return state

    slot = booking_request["slot"]

    try:
        from services.calendar import book_appointment
        from datetime import datetime

        slot_date = datetime.strptime(slot["date"], "%Y-%m-%d").date()
        slot_time_obj = datetime.strptime(slot["time"], "%H:%M").time()

        appointment = book_appointment(
            customer_id=customer_id,
            conversation_id=state.get("crm_result", {}).get("conversation_id") if state.get("crm_result") else None,
            slot_date=slot_date,
            slot_time=slot_time_obj,
            purpose="Product Demonstration"
        )

        if appointment:
            booking_request["appointment"] = appointment
            booking_request["appointment_id"] = appointment["id"]
            logger.info(f"Appointment created: {appointment['id']}")

            # Send confirmation email if customer email is available
            try:
                from services.mailer import send_appointment_email
                from services.crm import get_customer

                ctx = state.get("customer_context", {}) or {}
                to_email = ctx.get("email")
                cust_name = ctx.get("name")
                if not to_email or not cust_name:
                    db_cust = get_customer(customer_id) or {}
                    to_email = to_email or db_cust.get("email")
                    cust_name = cust_name or db_cust.get("name")

                email_status = send_appointment_email(to_email or "", cust_name or "", appointment)
                booking_request["email_status"] = email_status
            except Exception as mail_err:
                logger.warning(f"Appointment email error: {mail_err}")

            return {**state, "customer_id": customer_id, "booking_request": booking_request}
    except Exception as e:
        logger.error(f"Appointment creation error: {e}")

    return state


def node_update_crm(state: AgentState) -> AgentState:
    """Update CRM with customer data."""
    customer_context = state.get("customer_context", {}) or {}
    sentiment = state.get("sentiment", {}) or {}
    lead_score = state.get("lead_score", 0)
    qual_data = state.get("qualification_data", {}) or {}
    
    if not customer_context.get("name") and not customer_context.get("email"):
        return state  # Not enough data to create a CRM record
    
    try:
        from services.crm import create_or_update_customer, update_customer
        
        crm_data = {
            **customer_context,
            "lead_score": lead_score,
            "lead_category": qual_data.get("lead_category", "COLD"),
            "sentiment": sentiment.get("overall", "neutral"),
            "buying_intent": sentiment.get("buying_intent", "low"),
            "preferred_language": state.get("detected_language", "en"),
        }
        
        customer_id = state.get("customer_id")
        if customer_id:
            update_customer(customer_id, crm_data)
        else:
            customer_id = create_or_update_customer(crm_data)
        
        if customer_id:
            return {
                **state,
                "customer_id": customer_id,
                "crm_result": {"customer_id": customer_id, "status": "updated"}
            }
    except Exception as e:
        logger.error(f"CRM update error: {e}")
    
    return state


def node_generate_response(state: AgentState) -> AgentState:
    """Generate the final response to the customer."""
    messages = state.get("messages", [])
    if not messages:
        return {**state, "response": "Hello! How can I help you today?"}
    
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    intent = state.get("detected_intent", "general")
    language = state.get("detected_language", "en")
    customer_context = state.get("customer_context", {}) or {}
    rag_context = state.get("rag_context", "")
    retrieved_docs = state.get("retrieved_documents", [])
    available_slots = state.get("available_slots", [])
    booking_request = state.get("booking_request", {}) or {}
    history = state.get("conversation_history", [])
    
    response = ""
    
    # Handle escalation
    if state.get("should_escalate"):
        response = (
            "I completely understand your frustration, and I sincerely apologize for any inconvenience. "
            "I'd like to connect you with one of our senior sales managers who can better assist you. "
            "You can reach them directly at: 📧 sales@techvex.com or 📞 +91-80-4567-8900. "
            "Would you like me to arrange a callback?"
        )
        return {**state, "response": response}
    
    # Handle appointment confirmation success
    if booking_request.get("appointment"):
        appt = booking_request["appointment"]
        email_status = booking_request.get("email_status") or {}
        name = customer_context.get("name", "")
        greeting = f"Great news{', ' + name if name else ''}! "
        if email_status.get("sent"):
            email_note = f"I've sent the confirmation details to {email_status.get('to')}. "
        elif email_status.get("to"):
            email_note = f"Your appointment is saved under {email_status.get('to')}. "
        else:
            email_note = "What email address should I send the confirmation details to? "
        response = (
            f"{greeting}Your product demonstration has been confirmed for "
            f"{appt.get('datetime_display', 'the requested time')}. "
            f"{email_note}"
        )
        return {**state, "response": response}
    
    # Handle appointment slot suggestion
    if intent in ["appointment_request"] and available_slots:
        from services.calendar import format_slots_for_voice
        slots_text = format_slots_for_voice(available_slots)
        name = customer_context.get("name", "")
        response = (
            f"{'Of course' + (',' + name if name else '') + '!'} I'd be happy to schedule a product demonstration. "
            f"{slots_text} Which of these works best for you?"
        )
        return {**state, "response": response}
    
    # RAG-grounded response for product/pricing questions
    if retrieved_docs and rag_context:
        response = generate_rag_response(
            question=content,
            context=rag_context,
            customer_name=customer_context.get("name", ""),
            language=language
        )
        return {**state, "response": response}
    
    # General conversational response
    response = generate_conversational_response(
        message=content,
        history=history,
        customer_context=customer_context,
        intent=intent,
        language=language,
        rag_context=rag_context
    )
    
    return {**state, "response": response}


def node_save_conversation(state: AgentState) -> AgentState:
    """Save conversation message and update DB."""
    try:
        from services.crm import save_conversation, save_message
        
        session_id = state.get("session_id", "")
        customer_id = state.get("customer_id")
        language = state.get("detected_language", "en")
        sentiment = state.get("sentiment", {}) or {}
        intent = state.get("detected_intent", "general")
        response = state.get("response", "")
        messages = state.get("messages", [])
        
        # Ensure conversation exists
        conv_id = save_conversation(session_id, customer_id, language)
        
        if conv_id and messages:
            # Save last user message
            last_msg = messages[-1]
            user_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            save_message(conv_id, "user", user_content, language, sentiment.get("overall", "neutral"), intent)
            
            # Save agent response
            if response:
                save_message(conv_id, "assistant", response, language, "neutral", "response")
        
        # Update conversation history
        history = state.get("conversation_history", []) or []
        if messages:
            last_msg = messages[-1]
            user_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            history = history + [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response}
            ]
            # Bound history
            if len(history) > 40:
                history = history[-40:]
        
        crm_result = state.get("crm_result", {}) or {}
        crm_result["conversation_id"] = conv_id
        
        return {
            **state,
            "conversation_history": history,
            "crm_result": crm_result
        }
        
    except Exception as e:
        logger.error(f"Save conversation error: {e}")
        return state


# ─── CONDITIONAL ROUTING ─────────────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """Route after intent classification."""
    intent = state.get("detected_intent", "general")
    
    if intent in ["product_inquiry", "pricing_inquiry", "faq"]:
        return "retrieve_knowledge"
    elif intent in ["appointment_request", "appointment_confirmation"]:
        return "check_appointment_availability"
    elif intent in ["escalation_request"]:
        return "generate_response"
    else:
        return "extract_customer_information"


def route_after_extraction(state: AgentState) -> str:
    """Route after entity extraction."""
    intent = state.get("detected_intent", "general")
    
    if state.get("should_book"):
        return "check_appointment_availability"
    
    return "qualify_lead"


def route_after_rag(state: AgentState) -> str:
    """Route after RAG retrieval."""
    return "analyze_sentiment"


def route_to_crm(state: AgentState) -> str:
    """Route to CRM update if we have customer data."""
    context = state.get("customer_context", {}) or {}
    
    if context.get("name") or context.get("email") or context.get("company"):
        return "update_crm"
    
    return "generate_response"


# ─── BUILD GRAPH ─────────────────────────────────────────────────────────────

def build_agent_graph() -> Any:
    """Build and compile the LangGraph agent."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_customer_memory", node_load_customer_memory)
    workflow.add_node("detect_language", node_detect_language)
    workflow.add_node("classify_intent", node_classify_intent)
    workflow.add_node("retrieve_knowledge", node_retrieve_knowledge)
    workflow.add_node("extract_customer_information", node_extract_customer_information)
    workflow.add_node("qualify_lead", node_qualify_lead)
    workflow.add_node("analyze_sentiment", node_analyze_sentiment)
    workflow.add_node("check_appointment_availability", node_check_appointment_availability)
    workflow.add_node("confirm_appointment", node_confirm_appointment)
    workflow.add_node("create_appointment", node_create_appointment)
    workflow.add_node("update_crm", node_update_crm)
    workflow.add_node("generate_response", node_generate_response)
    workflow.add_node("save_conversation", node_save_conversation)
    
    # Entry point
    workflow.set_entry_point("load_customer_memory")
    
    # Linear first steps
    workflow.add_edge("load_customer_memory", "detect_language")
    workflow.add_edge("detect_language", "classify_intent")
    
    # Conditional routing after intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "retrieve_knowledge": "retrieve_knowledge",
            "check_appointment_availability": "check_appointment_availability",
            "extract_customer_information": "extract_customer_information",
            "generate_response": "generate_response",
        }
    )
    
    # RAG path
    workflow.add_edge("retrieve_knowledge", "analyze_sentiment")
    
    # Extraction path
    workflow.add_conditional_edges(
        "extract_customer_information",
        route_after_extraction,
        {
            "check_appointment_availability": "check_appointment_availability",
            "qualify_lead": "qualify_lead",
        }
    )
    
    workflow.add_edge("qualify_lead", "analyze_sentiment")
    
    # Appointment path
    workflow.add_edge("check_appointment_availability", "confirm_appointment")
    workflow.add_edge("confirm_appointment", "create_appointment")
    workflow.add_edge("create_appointment", "analyze_sentiment")
    
    # Merge at sentiment analysis
    workflow.add_conditional_edges(
        "analyze_sentiment",
        route_to_crm,
        {
            "update_crm": "update_crm",
            "generate_response": "generate_response",
        }
    )
    
    workflow.add_edge("update_crm", "generate_response")
    workflow.add_edge("generate_response", "save_conversation")
    workflow.add_edge("save_conversation", END)
    
    return workflow.compile()


# Global compiled graph instance
_agent_graph = None


def get_agent_graph():
    """Get or create the compiled agent graph."""
    global _agent_graph
    if _agent_graph is None:
        logger.info("Compiling LangGraph agent...")
        _agent_graph = build_agent_graph()
        logger.info("Agent graph compiled successfully.")
    return _agent_graph


def process_message(
    session_id: str,
    message: str,
    customer_id: Optional[str] = None,
    conversation_history: Optional[list] = None
) -> dict:
    """
    Process a single customer message through the agent graph.
    
    Args:
        session_id: Unique session identifier
        message: Customer's message text
        customer_id: Existing customer ID if known
        conversation_history: Previous conversation messages
    
    Returns:
        dict with response, lead_score, sentiment, etc.
    """
    from langchain_core.messages import HumanMessage
    
    graph = get_agent_graph()
    
    initial_state = {
        "session_id": session_id,
        "customer_id": customer_id,
        "messages": [HumanMessage(content=message)],
        "conversation_history": conversation_history or [],
        "customer_context": None,
        "detected_language": "en",
        "detected_intent": "general",
        "retrieved_documents": [],
        "rag_context": "",
        "qualification_data": None,
        "lead_score": 0.0,
        "sentiment": None,
        "booking_request": None,
        "available_slots": [],
        "crm_result": None,
        "response": "",
        "error_state": None,
        "next_action": "",
        "should_retrieve_rag": False,
        "should_qualify": True,
        "should_book": False,
        "should_escalate": False,
    }
    
    try:
        result = graph.invoke(initial_state)
        
        return {
            "response": result.get("response", "I apologize, could you repeat that?"),
            "session_id": session_id,
            "customer_id": result.get("customer_id"),
            "detected_language": result.get("detected_language", "en"),
            "detected_intent": result.get("detected_intent", "general"),
            "lead_score": result.get("lead_score", 0),
            "lead_category": result.get("qualification_data", {}).get("lead_category", "COLD") if result.get("qualification_data") else "COLD",
            "sentiment": result.get("sentiment", {"overall": "neutral"}),
            "customer_context": result.get("customer_context", {}),
            "retrieved_documents": result.get("retrieved_documents", []),
            "booking_request": result.get("booking_request", {}),
            "conversation_history": result.get("conversation_history", []),
            "qualification_data": result.get("qualification_data", {}),
        }
        
    except Exception as e:
        logger.error(f"Agent graph error: {e}", exc_info=True)
        return {
            "response": "I'm experiencing a technical issue. Please try again in a moment.",
            "session_id": session_id,
            "customer_id": customer_id,
            "error": str(e),
            "lead_score": 0,
            "lead_category": "COLD",
            "sentiment": {"overall": "neutral"},
            "customer_context": {},
            "conversation_history": conversation_history or [],
        }

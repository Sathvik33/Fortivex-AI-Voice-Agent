"""
LangGraph State definition for the Autonomous AI Sales Voice Agent.
Defines the typed state that flows through the agent graph.
"""

from typing import Any, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class CustomerContext(BaseModel):
    """Extracted customer information."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    company_size: Optional[int] = None
    budget: Optional[float] = None
    budget_currency: str = "INR"
    purchase_timeline: Optional[str] = None
    is_decision_maker: Optional[bool] = None
    requirements: Optional[str] = None
    objections: Optional[str] = None
    preferred_contact_time: Optional[str] = None
    product_interests: Optional[str] = None


class QualificationData(BaseModel):
    """Lead qualification scoring data."""
    budget_score: float = 0.0        # 0-25 points
    timeline_score: float = 0.0      # 0-20 points
    authority_score: float = 0.0     # 0-20 points
    requirement_score: float = 0.0   # 0-20 points
    engagement_score: float = 0.0    # 0-15 points
    total_score: float = 0.0         # 0-100 points
    lead_category: str = "COLD"      # HOT, WARM, COLD
    score_explanation: str = ""


class SentimentData(BaseModel):
    """Sentiment analysis results."""
    overall: str = "neutral"           # positive, neutral, negative
    buying_intent: str = "low"         # high, medium, low
    frustration_level: str = "none"    # high, medium, low, none
    objections_detected: list[str] = Field(default_factory=list)
    interest_level: str = "medium"     # high, medium, low
    urgency: str = "low"               # high, medium, low


class BookingRequest(BaseModel):
    """Appointment booking request."""
    requested: bool = False
    date_preference: Optional[str] = None
    time_preference: Optional[str] = None
    confirmed: bool = False
    appointment_id: Optional[str] = None


class AgentState(TypedDict):
    """Full agent state passed through the LangGraph graph."""
    # Session identification
    session_id: str
    customer_id: Optional[str]
    
    # Conversation
    messages: Annotated[list, add_messages]
    conversation_history: list[dict]
    
    # Customer data
    customer_context: Optional[dict]
    detected_language: str
    detected_intent: str
    
    # RAG
    retrieved_documents: list[dict]
    rag_context: str
    
    # Intelligence
    qualification_data: Optional[dict]
    lead_score: float
    sentiment: Optional[dict]
    
    # Booking
    booking_request: Optional[dict]
    available_slots: list[dict]
    
    # CRM
    crm_result: Optional[dict]
    
    # Response
    response: str
    error_state: Optional[str]
    
    # Control flow
    next_action: str
    should_retrieve_rag: bool
    should_qualify: bool
    should_book: bool
    should_escalate: bool

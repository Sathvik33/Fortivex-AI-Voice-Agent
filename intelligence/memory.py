"""
Intelligence module: Persistent customer memory management.
Manages short-term (session) and long-term (database) customer memory.
"""

import logging
import json
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CustomerMemory:
    """
    Manages customer memory across conversations.
    
    Short-term: Current session context (in-memory dict)
    Long-term: Database-persisted customer profile
    """
    
    def __init__(self, session_id: str, customer_id: Optional[str] = None):
        self.session_id = session_id
        self.customer_id = customer_id
        
        # Short-term memory
        self.context: dict = {
            "name": None,
            "email": None,
            "phone": None,
            "company": None,
            "company_size": None,
            "budget": None,
            "budget_currency": "INR",
            "purchase_timeline": None,
            "is_decision_maker": None,
            "requirements": None,
            "objections": None,
            "product_interests": None,
            "preferred_contact_time": None,
        }
        
        # Conversation state
        self.current_intent: str = "greeting"
        self.detected_language: str = "en"
        self.qualification_status: str = "not_started"
        self.sentiment_history: list = []
        self.topics_discussed: list = []
        
        # Load from DB if customer_id provided
        if customer_id:
            self._load_from_db(customer_id)
    
    def _load_from_db(self, customer_id: str):
        """Load existing customer data from database."""
        try:
            from services.crm import get_customer
            customer = get_customer(customer_id)
            if customer:
                self.context.update({
                    k: v for k, v in {
                        "name": customer.get("name"),
                        "email": customer.get("email"),
                        "phone": customer.get("phone"),
                        "company": customer.get("company"),
                        "company_size": customer.get("company_size"),
                        "budget": float(customer.get("budget")) if customer.get("budget") else None,
                        "budget_currency": customer.get("budget_currency", "INR"),
                        "purchase_timeline": customer.get("purchase_timeline"),
                        "is_decision_maker": customer.get("decision_maker"),
                        "requirements": customer.get("requirements"),
                        "objections": customer.get("objections"),
                    }.items() if v is not None
                })
                self.detected_language = customer.get("preferred_language", "en")
                logger.info(f"Loaded customer memory for: {customer.get('name', 'Unknown')}")
        except Exception as e:
            logger.warning(f"Could not load customer from DB: {e}")
    
    def update_context(self, extracted_data: dict):
        """Update context with newly extracted information."""
        updated_fields = []
        
        for key, value in extracted_data.items():
            if value is not None and value != "" and key in self.context:
                # Don't overwrite existing data with None/empty
                old_value = self.context.get(key)
                if old_value is None or (key == "requirements" and old_value):
                    # Append to requirements, overwrite others
                    if key == "requirements" and old_value:
                        self.context[key] = f"{old_value}; {value}"
                    else:
                        self.context[key] = value
                        updated_fields.append(key)
                else:
                    self.context[key] = value
                    updated_fields.append(key)
        
        if updated_fields:
            logger.info(f"Updated customer context: {updated_fields}")
        
        return updated_fields
    
    def get_context_summary(self) -> str:
        """Get a human-readable summary of known customer information."""
        parts = []
        
        if self.context.get("name"):
            parts.append(f"Name: {self.context['name']}")
        if self.context.get("company"):
            parts.append(f"Company: {self.context['company']}")
        if self.context.get("company_size"):
            parts.append(f"Size: {self.context['company_size']} employees")
        if self.context.get("budget"):
            currency = self.context.get("budget_currency", "INR")
            parts.append(f"Budget: {self.context['budget']:,.0f} {currency}")
        if self.context.get("purchase_timeline"):
            parts.append(f"Timeline: {self.context['purchase_timeline']}")
        if self.context.get("requirements"):
            parts.append(f"Requirements: {self.context['requirements']}")
        if self.context.get("is_decision_maker") is not None:
            parts.append(f"Decision maker: {'Yes' if self.context['is_decision_maker'] else 'No'}")
        
        return "\n".join(parts) if parts else "No customer information collected yet."
    
    def get_missing_fields(self) -> list[str]:
        """Get list of important missing fields."""
        important_fields = ["name", "company", "requirements", "budget", "purchase_timeline"]
        return [f for f in important_fields if not self.context.get(f)]
    
    def add_sentiment(self, sentiment: dict):
        """Track sentiment over time."""
        self.sentiment_history.append({
            "sentiment": sentiment.get("overall", "neutral"),
            "timestamp": datetime.now().isoformat()
        })
    
    def get_dominant_sentiment(self) -> str:
        """Get the most common sentiment from conversation."""
        if not self.sentiment_history:
            return "neutral"
        
        counts = {}
        for s in self.sentiment_history:
            sent = s["sentiment"]
            counts[sent] = counts.get(sent, 0) + 1
        
        return max(counts, key=counts.get)
    
    def to_dict(self) -> dict:
        """Serialize context to dict."""
        return {
            "session_id": self.session_id,
            "customer_id": self.customer_id,
            "context": self.context,
            "detected_language": self.detected_language,
            "qualification_status": self.qualification_status,
        }


class SessionStore:
    """In-memory session store for active conversations."""
    
    _sessions: dict = {}
    
    @classmethod
    def get_or_create(cls, session_id: str, customer_id: Optional[str] = None) -> CustomerMemory:
        """Get existing session or create new one."""
        if session_id not in cls._sessions:
            cls._sessions[session_id] = CustomerMemory(session_id, customer_id)
        return cls._sessions[session_id]
    
    @classmethod
    def get(cls, session_id: str) -> Optional[CustomerMemory]:
        """Get existing session."""
        return cls._sessions.get(session_id)
    
    @classmethod
    def end_session(cls, session_id: str):
        """End and remove a session."""
        cls._sessions.pop(session_id, None)
    
    @classmethod
    def active_sessions(cls) -> int:
        """Count active sessions."""
        return len(cls._sessions)

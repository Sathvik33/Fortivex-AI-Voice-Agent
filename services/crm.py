"""
Services module: CRM operations.
Create, read, update, search customer records in NeonDB.
"""

import uuid
import logging
from typing import Optional
from datetime import datetime

from database.db import execute_query, get_db

logger = logging.getLogger(__name__)


def create_or_update_customer(customer_data: dict) -> Optional[str]:
    """
    Create a new customer or update existing one (by email).
    Returns customer_id.
    """
    email = customer_data.get("email")
    
    # Try to find existing customer by email
    if email:
        existing = get_customer_by_email(email)
        if existing:
            customer_id = str(existing["id"])
            update_customer(customer_id, customer_data)
            return customer_id
    
    # Create new customer
    return create_customer(customer_data)


def create_customer(data: dict) -> Optional[str]:
    """Create a new customer record."""
    try:
        result = execute_query(
            """
            INSERT INTO customers (
                name, email, phone, company, company_size, budget, budget_currency,
                requirements, lead_score, lead_category, sentiment, preferred_language,
                last_interaction, follow_up_status, decision_maker, purchase_timeline,
                objections, buying_intent
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                NOW(), %s, %s, %s,
                %s, %s
            )
            RETURNING id
            """,
            (
                data.get("name"),
                data.get("email"),
                data.get("phone"),
                data.get("company"),
                data.get("company_size"),
                data.get("budget"),
                data.get("budget_currency", "INR"),
                data.get("requirements"),
                data.get("lead_score", 0),
                data.get("lead_category", "COLD"),
                data.get("sentiment", "neutral"),
                data.get("preferred_language", "en"),
                data.get("follow_up_status", "pending"),
                data.get("is_decision_maker"),
                data.get("purchase_timeline"),
                data.get("objections"),
                data.get("buying_intent", "low")
            ),
            fetch_one=True
        )
        
        customer_id = str(result["id"]) if result else None
        logger.info(f"Created customer: {customer_id} ({data.get('name', 'Unknown')})")
        return customer_id
        
    except Exception as e:
        logger.error(f"Failed to create customer: {e}")
        return None


def update_customer(customer_id: str, data: dict) -> bool:
    """Update customer record with new data (only provided fields)."""
    if not data:
        return False
    
    # Build dynamic update query
    fields = []
    values = []
    
    field_map = {
        "name": "name",
        "email": "email",
        "phone": "phone",
        "company": "company",
        "company_size": "company_size",
        "budget": "budget",
        "budget_currency": "budget_currency",
        "requirements": "requirements",
        "lead_score": "lead_score",
        "lead_category": "lead_category",
        "sentiment": "sentiment",
        "preferred_language": "preferred_language",
        "follow_up_status": "follow_up_status",
        "is_decision_maker": "decision_maker",
        "purchase_timeline": "purchase_timeline",
        "objections": "objections",
        "buying_intent": "buying_intent",
    }
    
    for data_key, db_col in field_map.items():
        if data.get(data_key) is not None:
            fields.append(f"{db_col} = %s")
            values.append(data[data_key])
    
    if not fields:
        return False
    
    # Always update last_interaction and updated_at
    fields.append("last_interaction = NOW()")
    fields.append("updated_at = NOW()")
    values.append(customer_id)
    
    try:
        execute_query(
            f"UPDATE customers SET {', '.join(fields)} WHERE id = %s::uuid",
            tuple(values)
        )
        logger.info(f"Updated customer: {customer_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to update customer {customer_id}: {e}")
        return False


def get_customer_by_email(email: str) -> Optional[dict]:
    """Retrieve customer by email (case-insensitive)."""
    try:
        result = execute_query(
            "SELECT * FROM customers WHERE LOWER(email) = LOWER(%s)",
            (email,),
            fetch_one=True
        )
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get customer by email: {e}")
        return None


def get_customer_by_name(name: str) -> Optional[dict]:
    """Retrieve most recent customer by exact name (case-insensitive)."""
    try:
        result = execute_query(
            "SELECT * FROM customers WHERE LOWER(name) = LOWER(%s) ORDER BY updated_at DESC LIMIT 1",
            (name,),
            fetch_one=True
        )
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get customer by name: {e}")
        return None


def get_customer(customer_id: str) -> Optional[dict]:
    """Retrieve customer by ID."""
    try:
        result = execute_query(
            "SELECT * FROM customers WHERE id = %s::uuid",
            (customer_id,),
            fetch_one=True
        )
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get customer {customer_id}: {e}")
        return None


def search_customers(
    query: str = "",
    lead_category: str = None,
    sentiment: str = None,
    limit: int = 50,
    offset: int = 0
) -> list[dict]:
    """Search customers with filters."""
    conditions = []
    params = []
    
    if query:
        conditions.append(
            "(name ILIKE %s OR email ILIKE %s OR company ILIKE %s)"
        )
        params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
    
    if lead_category:
        conditions.append("lead_category = %s")
        params.append(lead_category)
    
    if sentiment:
        conditions.append("sentiment = %s")
        params.append(sentiment)
    
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.extend([limit, offset])
    
    try:
        rows = execute_query(
            f"""
            SELECT * FROM customers
            {where}
            ORDER BY last_interaction DESC NULLS LAST, created_at DESC
            LIMIT %s OFFSET %s
            """,
            tuple(params),
            fetch_all=True
        )
        return [dict(row) for row in rows] if rows else []
    except Exception as e:
        logger.error(f"Customer search failed: {e}")
        return []


def get_all_customers(limit: int = 100) -> list[dict]:
    """Get all customers."""
    return search_customers(limit=limit)


def save_conversation(
    session_id: str,
    customer_id: Optional[str],
    detected_language: str = "en",
    status: str = "active"
) -> Optional[str]:
    """Create or get conversation record."""
    try:
        # Try to get existing
        existing = execute_query(
            "SELECT id FROM conversations WHERE session_id = %s",
            (session_id,),
            fetch_one=True
        )
        
        if existing:
            return str(existing["id"])
        
        result = execute_query(
            """
            INSERT INTO conversations (session_id, customer_id, detected_language, status)
            VALUES (%s, %s::uuid, %s, %s)
            RETURNING id
            """,
            (session_id, customer_id, detected_language, status),
            fetch_one=True
        )
        
        return str(result["id"]) if result else None
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        return None


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    detected_language: str = "en",
    sentiment: str = "neutral",
    intent: str = ""
) -> bool:
    """Save a conversation message."""
    try:
        execute_query(
            """
            INSERT INTO messages (conversation_id, role, content, detected_language, sentiment, intent)
            VALUES (%s::uuid, %s, %s, %s, %s, %s)
            """,
            (conversation_id, role, content, detected_language, sentiment, intent)
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        return False


def end_conversation(conversation_id: str, final_lead_score: float = 0, final_sentiment: str = "neutral") -> bool:
    """Mark conversation as ended."""
    try:
        execute_query(
            """
            UPDATE conversations 
            SET end_time = NOW(), final_lead_score = %s, final_sentiment = %s, status = 'completed'
            WHERE id = %s::uuid
            """,
            (final_lead_score, final_sentiment, conversation_id)
        )
        return True
    except Exception as e:
        logger.error(f"Failed to end conversation: {e}")
        return False


def get_conversation_messages(conversation_id: str) -> list[dict]:
    """Get all messages for a conversation."""
    try:
        rows = execute_query(
            """
            SELECT role, content, detected_language, sentiment, intent, timestamp
            FROM messages
            WHERE conversation_id = %s::uuid
            ORDER BY timestamp ASC
            """,
            (conversation_id,),
            fetch_all=True
        )
        return [dict(row) for row in rows] if rows else []
    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        return []

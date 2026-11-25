"""
Services module: Appointment/Calendar management.
Handles slot availability, booking, confirmation, and cancellation.
Uses SQL transactions to prevent double-booking.
"""

import logging
from typing import Optional
from datetime import datetime, date, time, timedelta

from database.db import execute_query, get_db

logger = logging.getLogger(__name__)


def get_available_slots(
    from_date: Optional[date] = None,
    days_ahead: int = 7,
    limit: int = 10
) -> list[dict]:
    """
    Get available appointment slots.
    
    Args:
        from_date: Start date (defaults to tomorrow)
        days_ahead: How many days to look ahead
        limit: Max slots to return
    
    Returns:
        List of available slot dicts
    """
    if from_date is None:
        from_date = date.today() + timedelta(days=1)
    
    to_date = from_date + timedelta(days=days_ahead)
    
    try:
        rows = execute_query(
            """
            SELECT s.id, s.slot_date, s.slot_time, s.is_available
            FROM available_slots s
            WHERE s.is_available = TRUE
                AND s.slot_date >= %s
                AND s.slot_date <= %s
                AND NOT EXISTS (
                    SELECT 1 FROM appointments a
                    WHERE a.appointment_date = s.slot_date
                        AND a.appointment_time = s.slot_time
                        AND a.status != 'cancelled'
                )
            ORDER BY s.slot_date, s.slot_time
            LIMIT %s
            """,
            (from_date, to_date, limit),
            fetch_all=True
        )
        
        slots = []
        for row in (rows or []):
            slot_dt = datetime.combine(row["slot_date"], row["slot_time"])
            slots.append({
                "id": str(row["id"]),
                "date": row["slot_date"].strftime("%Y-%m-%d"),
                "time": row["slot_time"].strftime("%H:%M"),
                "datetime_str": slot_dt.strftime("%A, %B %d at %I:%M %p"),
                "day_name": slot_dt.strftime("%A"),
                "display": slot_dt.strftime("%d %b %Y at %I:%M %p"),
            })
        
        return slots
        
    except Exception as e:
        logger.error(f"Failed to get available slots: {e}")
        return []


def format_slots_for_voice(slots: list[dict]) -> str:
    """Format available slots for voice response."""
    if not slots:
        return "No slots available for the next week."
    
    slot_texts = []
    for i, slot in enumerate(slots[:3], 1):  # Only mention 3 slots in voice
        slot_texts.append(f"Option {i}: {slot['datetime_str']}")
    
    return "Here are the available slots: " + ". ".join(slot_texts)


def book_appointment(
    customer_id: str,
    conversation_id: Optional[str],
    slot_date: date,
    slot_time: time,
    purpose: str = "Product Demonstration",
    notes: str = ""
) -> Optional[dict]:
    """
    Book an appointment with transaction protection.
    
    Returns:
        Appointment dict on success, None on failure (slot taken)
    """
    with get_db() as conn:
        try:
            with conn.cursor() as cur:
                # Lock the slot row for update (prevents race conditions)
                cur.execute(
                    """
                    SELECT id FROM available_slots
                    WHERE slot_date = %s AND slot_time = %s AND is_available = TRUE
                    FOR UPDATE NOWAIT
                    """,
                    (slot_date, slot_time)
                )
                
                slot_row = cur.fetchone()
                if not slot_row:
                    logger.warning(f"Slot not available: {slot_date} {slot_time}")
                    return None
                
                # Check no existing appointment (extra safety)
                cur.execute(
                    """
                    SELECT id FROM appointments
                    WHERE appointment_date = %s AND appointment_time = %s
                        AND status != 'cancelled'
                    """,
                    (slot_date, slot_time)
                )
                
                if cur.fetchone():
                    logger.warning(f"Appointment conflict: {slot_date} {slot_time}")
                    return None
                
                # Create appointment
                cur.execute(
                    """
                    INSERT INTO appointments (
                        customer_id, conversation_id, appointment_date, appointment_time,
                        purpose, status, confirmed_by_customer, notes
                    )
                    VALUES (%s::uuid, %s::uuid, %s, %s, %s, 'confirmed', TRUE, %s)
                    RETURNING id, appointment_date, appointment_time
                    """,
                    (
                        customer_id, conversation_id, slot_date, slot_time,
                        purpose, notes
                    )
                )
                
                appt = cur.fetchone()
                
                # Mark slot as booked
                cur.execute(
                    "UPDATE available_slots SET is_available = FALSE WHERE slot_date = %s AND slot_time = %s",
                    (slot_date, slot_time)
                )
                
                conn.commit()
                
                appt_dt = datetime.combine(slot_date, slot_time)
                
                result = {
                    "id": str(appt["id"]),
                    "date": slot_date.strftime("%Y-%m-%d"),
                    "time": slot_time.strftime("%H:%M"),
                    "datetime_display": appt_dt.strftime("%A, %B %d, %Y at %I:%M %p"),
                    "purpose": purpose,
                    "status": "confirmed"
                }
                
                logger.info(f"Appointment booked: {result['id']} for customer {customer_id}")
                return result
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Booking failed: {e}")
            return None


def has_explicit_slot_choice(text: str) -> bool:
    """Return True if the user message specifies a time, day, option, or confirmation."""
    import re
    t = text.lower().replace("a.m.", "am").replace("p.m.", "pm")
    if re.search(r'\b(9|10|11|12|1|2|3|4|5|6|14|15|16|17)\s*(?::\d{2})?\s*(am|pm)\b', t):
        return True
    if re.search(r'\boption\s*[1-3]\b', t):
        return True
    if any(w in t for w in ["yes", "confirm", "book it", "that works", "sounds good", "go ahead", "haan", "हाँ"]):
        return True
    return False


def parse_slot_preference(preference: str, available_slots: list[dict]) -> Optional[dict]:
    """
    Parse customer's slot preference and find best matching slot.
    
    Args:
        preference: Natural language preference e.g. "tomorrow 3pm", "9 a.m.", "option 1"
        available_slots: List of available slots
    
    Returns:
        Best matching slot dict or None
    """
    import re
    if not available_slots:
        return None
    
    preference_lower = preference.lower().strip().replace("a.m.", "am").replace("p.m.", "pm")

    # Option number matching (Option 1 / 2 / 3)
    opt_match = re.search(r'\boption\s*([1-6])\b', preference_lower)
    if opt_match:
        idx = int(opt_match.group(1)) - 1
        if 0 <= idx < len(available_slots):
            return available_slots[idx]
    
    # Time-based matching with regex
    hour_to_24 = {
        "9": "09:00", "10": "10:00", "11": "11:00",
        "14": "14:00", "2": "14:00",
        "15": "15:00", "3": "15:00",
        "16": "16:00", "4": "16:00",
        "17": "17:00", "5": "17:00",
    }
    
    target_time = None
    m = re.search(r'\b(1[0-7]|[1-9])\s*(?::\d{2})?\s*(am|pm)?\b', preference_lower)
    if m and m.group(1) in hour_to_24:
        target_time = hour_to_24[m.group(1)]
    
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "tomorrow": -1, "today": -2,
    }
    
    # Try to match day
    target_day = None
    today = date.today()
    for day_name, day_offset in day_map.items():
        if day_name in preference_lower:
            if day_offset == -1:  # tomorrow
                target_day = today + timedelta(days=1)
            elif day_offset >= 0:  # day of week
                days_ahead = (day_offset - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target_day = today + timedelta(days=days_ahead)
            break
    
    # Find best match
    for slot in available_slots:
        slot_date = datetime.strptime(slot["date"], "%Y-%m-%d").date()
        slot_time_str = slot["time"]
        
        date_match = target_day is None or slot_date == target_day
        time_match = target_time is None or slot_time_str == target_time
        
        if date_match and time_match:
            return slot
    
    # Fallback: return first available
    return available_slots[0] if available_slots else None


def cancel_appointment(appointment_id: str) -> bool:
    """Cancel an appointment and free the slot."""
    with get_db() as conn:
        try:
            with conn.cursor() as cur:
                # Get appointment details
                cur.execute(
                    "SELECT appointment_date, appointment_time FROM appointments WHERE id = %s::uuid",
                    (appointment_id,)
                )
                appt = cur.fetchone()
                
                if not appt:
                    return False
                
                # Cancel appointment
                cur.execute(
                    "UPDATE appointments SET status = 'cancelled', updated_at = NOW() WHERE id = %s::uuid",
                    (appointment_id,)
                )
                
                # Free the slot
                cur.execute(
                    "UPDATE available_slots SET is_available = TRUE WHERE slot_date = %s AND slot_time = %s",
                    (appt["appointment_date"], appt["appointment_time"])
                )
                
                conn.commit()
                logger.info(f"Cancelled appointment: {appointment_id}")
                return True
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to cancel appointment {appointment_id}: {e}")
            return False


def get_appointments(customer_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    """Get appointments with optional customer filter."""
    try:
        if customer_id:
            rows = execute_query(
                """
                SELECT a.*, c.name as customer_name, c.company
                FROM appointments a
                LEFT JOIN customers c ON a.customer_id = c.id
                WHERE a.customer_id = %s::uuid
                ORDER BY a.appointment_date DESC, a.appointment_time DESC
                LIMIT %s
                """,
                (customer_id, limit),
                fetch_all=True
            )
        else:
            rows = execute_query(
                """
                SELECT a.*, c.name as customer_name, c.company
                FROM appointments a
                LEFT JOIN customers c ON a.customer_id = c.id
                ORDER BY a.appointment_date DESC, a.appointment_time DESC
                LIMIT %s
                """,
                (limit,),
                fetch_all=True
            )
        
        return [dict(row) for row in rows] if rows else []
    except Exception as e:
        logger.error(f"Failed to get appointments: {e}")
        return []

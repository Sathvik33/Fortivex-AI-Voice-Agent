"""
Services module: Integration adapters.
Clean adapter pattern so external CRMs/calendars can be added later
without rewriting the agent logic.

Currently provides LOCAL (NeonDB) implementations.
HubSpot and Google Calendar are DOCUMENTED but NOT connected.
"""

import logging
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CRMAdapter(ABC):
    """Abstract CRM adapter interface."""
    
    @abstractmethod
    def create_contact(self, data: dict) -> Optional[str]:
        pass
    
    @abstractmethod
    def update_contact(self, contact_id: str, data: dict) -> bool:
        pass
    
    @abstractmethod
    def get_contact(self, contact_id: str) -> Optional[dict]:
        pass
    
    @abstractmethod
    def search_contacts(self, query: str) -> list[dict]:
        pass


class CalendarAdapter(ABC):
    """Abstract calendar adapter interface."""
    
    @abstractmethod
    def get_available_slots(self, days_ahead: int = 7) -> list[dict]:
        pass
    
    @abstractmethod
    def book_slot(self, slot_id: str, customer_data: dict) -> Optional[dict]:
        pass
    
    @abstractmethod
    def cancel_booking(self, booking_id: str) -> bool:
        pass


class LocalCRMAdapter(CRMAdapter):
    """
    LOCAL CRM implementation using NeonDB.
    This is the active implementation.
    """
    
    def create_contact(self, data: dict) -> Optional[str]:
        from services.crm import create_or_update_customer
        return create_or_update_customer(data)
    
    def update_contact(self, contact_id: str, data: dict) -> bool:
        from services.crm import update_customer
        return update_customer(contact_id, data)
    
    def get_contact(self, contact_id: str) -> Optional[dict]:
        from services.crm import get_customer
        return get_customer(contact_id)
    
    def search_contacts(self, query: str) -> list[dict]:
        from services.crm import search_customers
        return search_customers(query)


class LocalCalendarAdapter(CalendarAdapter):
    """
    LOCAL calendar implementation using NeonDB.
    This is the active implementation.
    """
    
    def get_available_slots(self, days_ahead: int = 7) -> list[dict]:
        from services.calendar import get_available_slots
        return get_available_slots(days_ahead=days_ahead)
    
    def book_slot(self, customer_id: str, slot_date, slot_time, purpose: str = "Demo") -> Optional[dict]:
        from services.calendar import book_appointment
        return book_appointment(customer_id, None, slot_date, slot_time, purpose)
    
    def cancel_booking(self, booking_id: str) -> bool:
        from services.calendar import cancel_appointment
        return cancel_appointment(booking_id)


class HubSpotCRMAdapter(CRMAdapter):
    """
    HubSpot CRM adapter — NOT CONNECTED.
    
    To connect: 
    1. Install hubspot-api-client
    2. Set HUBSPOT_API_KEY in .env
    3. Implement the methods below
    
    This is a future production integration path.
    """
    
    def __init__(self):
        logger.warning("HubSpot CRM adapter is NOT connected. Using local CRM.")
    
    def create_contact(self, data: dict) -> Optional[str]:
        raise NotImplementedError("HubSpot integration not connected")
    
    def update_contact(self, contact_id: str, data: dict) -> bool:
        raise NotImplementedError("HubSpot integration not connected")
    
    def get_contact(self, contact_id: str) -> Optional[dict]:
        raise NotImplementedError("HubSpot integration not connected")
    
    def search_contacts(self, query: str) -> list[dict]:
        raise NotImplementedError("HubSpot integration not connected")


class GoogleCalendarAdapter(CalendarAdapter):
    """
    Google Calendar adapter — NOT CONNECTED.
    
    To connect:
    1. Install google-auth google-auth-oauthlib google-api-python-client
    2. Set GOOGLE_CREDENTIALS_FILE in .env
    3. Implement OAuth2 flow
    4. Implement calendar API calls
    
    This is a future production integration path.
    """
    
    def __init__(self):
        logger.warning("Google Calendar adapter is NOT connected. Using local calendar.")
    
    def get_available_slots(self, days_ahead: int = 7) -> list[dict]:
        raise NotImplementedError("Google Calendar integration not connected")
    
    def book_slot(self, customer_id: str, slot_date, slot_time, purpose: str = "Demo") -> Optional[dict]:
        raise NotImplementedError("Google Calendar integration not connected")
    
    def cancel_booking(self, booking_id: str) -> bool:
        raise NotImplementedError("Google Calendar integration not connected")


# Active adapters — swap these to switch implementations
def get_crm_adapter() -> CRMAdapter:
    """Get the active CRM adapter."""
    return LocalCRMAdapter()


def get_calendar_adapter() -> CalendarAdapter:
    """Get the active calendar adapter."""
    return LocalCalendarAdapter()

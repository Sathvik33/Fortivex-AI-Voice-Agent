"""
Services module: Email notifications via 3rd-party HTTP APIs (Resend, Brevo)
with optional SMTP fallback. No local mail server required.
"""

import os
import smtplib
import logging
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def _build_email_content(customer_name: str, appointment: dict) -> tuple[str, str, str]:
    """Build subject, plain-text body, and HTML body for an appointment confirmation."""
    name = customer_name or "Valued Customer"
    dt_display = appointment.get("datetime_display") or f"{appointment.get('date')} at {appointment.get('time')}"
    purpose = appointment.get("purpose", "Product Demonstration")
    appt_id = appointment.get("id", "")

    subject = f"Appointment Confirmed: {purpose} on {dt_display} — TechVex Solutions"

    plain_body = (
        f"Hello {name},\n\n"
        f"Your appointment with TechVex Solutions is confirmed.\n\n"
        f"Details:\n"
        f"- Purpose: {purpose}\n"
        f"- Date & Time: {dt_display}\n"
        f"- Reference ID: {appt_id}\n\n"
        f"If you need to reschedule or have questions beforehand, reply to this email or call +91-80-4567-8900.\n\n"
        f"Best regards,\n"
        f"TechVex Solutions Sales Team\n"
    )

    html_body = f"""
    <div style="font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 560px; margin: 0 auto; color: #0f172a; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden;">
      <div style="background: #0f172a; color: #ffffff; padding: 20px 24px;">
        <div style="font-size: 16px; font-weight: 600;">TechVex Solutions</div>
        <div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">Appointment Confirmation</div>
      </div>
      <div style="padding: 24px;">
        <p style="margin: 0 0 14px;">Hello <strong>{name}</strong>,</p>
        <p style="margin: 0 0 18px; color: #475569; font-size: 14px;">Your product demonstration has been scheduled. Here are your appointment details:</p>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
          <tr><td style="padding: 10px 14px; color: #64748b;">Purpose</td><td style="padding: 10px 14px; font-weight: 600;">{purpose}</td></tr>
          <tr><td style="padding: 10px 14px; color: #64748b; border-top: 1px solid #e2e8f0;">Date &amp; Time</td><td style="padding: 10px 14px; font-weight: 600; border-top: 1px solid #e2e8f0;">{dt_display}</td></tr>
          <tr><td style="padding: 10px 14px; color: #64748b; border-top: 1px solid #e2e8f0;">Reference ID</td><td style="padding: 10px 14px; font-family: monospace; border-top: 1px solid #e2e8f0;">{appt_id}</td></tr>
        </table>
        <p style="margin: 18px 0 0; font-size: 12px; color: #64748b;">
          Questions? Contact us at sales@techvex.com or +91-80-4567-8900.
        </p>
      </div>
    </div>
    """
    return subject, plain_body, html_body


def _send_via_resend(to_email: str, subject: str, plain_body: str, html_body: str, api_key: str) -> dict:
    """Send email via Resend REST API (https://api.resend.com/emails)."""
    from_addr = os.getenv("EMAIL_FROM", "TechVex Sales <onboarding@resend.dev>")
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": from_addr,
            "to": [to_email],
            "subject": subject,
            "text": plain_body,
            "html": html_body,
        },
        timeout=15,
    )
    if resp.status_code in (200, 201):
        logger.info(f"Resend email sent to {to_email}: {resp.json().get('id')}")
        return {"sent": True, "status": "sent", "provider": "resend", "to": to_email}
    logger.error(f"Resend error ({resp.status_code}): {resp.text[:200]}")
    return {"sent": False, "status": "error", "provider": "resend", "to": to_email, "detail": resp.text[:200]}


def _send_via_brevo(to_email: str, customer_name: str, subject: str, plain_body: str, html_body: str, api_key: str) -> dict:
    """Send email via Brevo REST API (https://api.brevo.com/v3/smtp/email)."""
    from_email = os.getenv("EMAIL_FROM", "sales@techvex.com")
    resp = requests.post(
        "https://api.brevo.com/v3/smtp/email",
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "sender": {"name": "TechVex Solutions", "email": from_email},
            "to": [{"email": to_email, "name": customer_name or to_email}],
            "subject": subject,
            "textContent": plain_body,
            "htmlContent": html_body,
        },
        timeout=15,
    )
    if resp.status_code in (200, 201):
        logger.info(f"Brevo email sent to {to_email}")
        return {"sent": True, "status": "sent", "provider": "brevo", "to": to_email}
    logger.error(f"Brevo error ({resp.status_code}): {resp.text[:200]}")
    return {"sent": False, "status": "error", "provider": "brevo", "to": to_email, "detail": resp.text[:200]}


def send_appointment_email(
    to_email: str,
    customer_name: str,
    appointment: dict,
) -> dict:
    """
    Send appointment confirmation details to the customer's email.
    Priority:
      1. Resend API (RESEND_API_KEY)
      2. Brevo API (BREVO_API_KEY)
      3. Optional SMTP fallback (SMTP_USER / SMTP_PASSWORD)
    """
    if not to_email or "@" not in to_email:
        return {"sent": False, "status": "no_recipient", "detail": "No valid customer email on file."}

    subject, plain_body, html_body = _build_email_content(customer_name, appointment)

    resend_key = os.getenv("RESEND_API_KEY", "").strip()
    if resend_key:
        try:
            return _send_via_resend(to_email, subject, plain_body, html_body, resend_key)
        except Exception as e:
            logger.error(f"Resend exception: {e}")
            return {"sent": False, "status": "error", "provider": "resend", "to": to_email, "detail": str(e)}

    brevo_key = os.getenv("BREVO_API_KEY", "").strip()
    if brevo_key:
        try:
            return _send_via_brevo(to_email, customer_name, subject, plain_body, html_body, brevo_key)
        except Exception as e:
            logger.error(f"Brevo exception: {e}")
            return {"sent": False, "status": "error", "provider": "brevo", "to": to_email, "detail": str(e)}

    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    if smtp_user and smtp_password:
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_from = os.getenv("SMTP_FROM", smtp_user)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_email
        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            return {"sent": True, "status": "sent", "provider": "smtp", "to": to_email}
        except Exception as e:
            return {"sent": False, "status": "error", "provider": "smtp", "to": to_email, "detail": str(e)}

    logger.info(f"No email provider key configured; skipping email to {to_email}")
    return {
        "sent": False,
        "status": "email_not_configured",
        "to": to_email,
        "detail": "Set RESEND_API_KEY or BREVO_API_KEY in .env to enable 3rd-party email delivery.",
    }

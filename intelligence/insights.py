"""
Intelligence module: Business insights and analytics.
Aggregates data from NeonDB for dashboard display.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_dashboard_kpis() -> dict:
    """Get key performance indicators for the dashboard."""
    from database.db import execute_query
    
    try:
        # Total calls
        total_calls = execute_query(
            "SELECT COUNT(*) as count FROM conversations",
            fetch_one=True
        )
        
        # Total unique customers
        total_customers = execute_query(
            "SELECT COUNT(*) as count FROM customers",
            fetch_one=True
        )
        
        # Lead category breakdown
        lead_counts = execute_query(
            """
            SELECT lead_category, COUNT(*) as count 
            FROM customers 
            GROUP BY lead_category
            """,
            fetch_all=True
        )
        
        lead_map = {row["lead_category"]: int(row["count"]) for row in (lead_counts or [])}
        
        # Appointments
        appt_stats = execute_query(
            """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'confirmed' THEN 1 END) as confirmed,
                COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled
            FROM appointments
            """,
            fetch_one=True
        )
        
        # Average lead score
        avg_score = execute_query(
            "SELECT AVG(lead_score) as avg FROM customers WHERE lead_score > 0",
            fetch_one=True
        )
        
        # Sentiment distribution
        sentiment_counts = execute_query(
            """
            SELECT sentiment, COUNT(*) as count
            FROM customers
            GROUP BY sentiment
            """,
            fetch_all=True
        )
        sentiment_map = {row["sentiment"]: int(row["count"]) for row in (sentiment_counts or [])}
        
        total = int(total_calls["count"]) if total_calls else 0
        confirmed_appts = int(appt_stats["confirmed"]) if appt_stats else 0
        conversion_rate = (confirmed_appts / total * 100) if total > 0 else 0
        
        return {
            "total_calls": total,
            "total_customers": int(total_customers["count"]) if total_customers else 0,
            "hot_leads": lead_map.get("HOT", 0),
            "warm_leads": lead_map.get("WARM", 0),
            "cold_leads": lead_map.get("COLD", 0),
            "total_qualified": lead_map.get("HOT", 0) + lead_map.get("WARM", 0),
            "appointments_booked": int(appt_stats["total"]) if appt_stats else 0,
            "appointments_confirmed": confirmed_appts,
            "appointment_conversion_rate": round(conversion_rate, 1),
            "avg_lead_score": round(float(avg_score["avg"]) if avg_score and avg_score["avg"] else 0, 1),
            "sentiment_positive": sentiment_map.get("positive", 0),
            "sentiment_neutral": sentiment_map.get("neutral", 0),
            "sentiment_negative": sentiment_map.get("negative", 0),
        }
    except Exception as e:
        logger.error(f"Failed to get KPIs: {e}")
        return {
            "total_calls": 0, "total_customers": 0,
            "hot_leads": 0, "warm_leads": 0, "cold_leads": 0,
            "total_qualified": 0, "appointments_booked": 0,
            "appointments_confirmed": 0, "appointment_conversion_rate": 0,
            "avg_lead_score": 0, "sentiment_positive": 0,
            "sentiment_neutral": 0, "sentiment_negative": 0,
        }


def get_lead_distribution_data() -> dict:
    """Get lead distribution data for Plotly chart."""
    from database.db import execute_query
    
    try:
        rows = execute_query(
            """
            SELECT lead_category, COUNT(*) as count
            FROM customers
            GROUP BY lead_category
            ORDER BY count DESC
            """,
            fetch_all=True
        )
        
        if not rows:
            return {"labels": ["HOT", "WARM", "COLD"], "values": [0, 0, 0]}
        
        labels = [row["lead_category"] for row in rows]
        values = [int(row["count"]) for row in rows]
        
        return {"labels": labels, "values": values}
    except Exception as e:
        logger.error(f"Failed to get lead distribution: {e}")
        return {"labels": [], "values": []}


def get_sentiment_data() -> dict:
    """Get sentiment distribution for Plotly chart."""
    from database.db import execute_query
    
    try:
        rows = execute_query(
            """
            SELECT sentiment, COUNT(*) as count
            FROM customers
            GROUP BY sentiment
            """,
            fetch_all=True
        )
        
        if not rows:
            return {"labels": ["positive", "neutral", "negative"], "values": [0, 0, 0]}
        
        return {
            "labels": [row["sentiment"] for row in rows],
            "values": [int(row["count"]) for row in rows]
        }
    except Exception as e:
        logger.error(f"Failed to get sentiment data: {e}")
        return {"labels": [], "values": []}


def get_calls_over_time(days: int = 30) -> dict:
    """Get call volume over time for trend chart."""
    from database.db import execute_query
    
    try:
        from_date = datetime.now() - timedelta(days=days)
        rows = execute_query(
            """
            SELECT DATE(start_time) as date, COUNT(*) as count
            FROM conversations
            WHERE start_time >= %s
            GROUP BY DATE(start_time)
            ORDER BY date ASC
            """,
            (from_date,),
            fetch_all=True
        )
        
        if not rows:
            return {"dates": [], "counts": []}
        
        return {
            "dates": [str(row["date"]) for row in rows],
            "counts": [int(row["count"]) for row in rows]
        }
    except Exception as e:
        logger.error(f"Failed to get calls over time: {e}")
        return {"dates": [], "counts": []}


def get_recent_summaries(limit: int = 10) -> list[dict]:
    """Get recent call summaries."""
    from database.db import execute_query
    
    try:
        rows = execute_query(
            """
            SELECT 
                cs.*, 
                cs.created_at::text as created_at_str
            FROM call_summaries cs
            ORDER BY cs.created_at DESC
            LIMIT %s
            """,
            (limit,),
            fetch_all=True
        )
        return [dict(row) for row in rows] if rows else []
    except Exception as e:
        logger.error(f"Failed to get recent summaries: {e}")
        return []


def get_follow_up_recommendations() -> list[dict]:
    """Get prioritized follow-up recommendations."""
    from database.db import execute_query
    
    try:
        rows = execute_query(
            """
            SELECT 
                c.id, c.name, c.company, c.email, c.lead_score, c.lead_category,
                c.sentiment, c.last_interaction, c.follow_up_status,
                c.budget, c.purchase_timeline
            FROM customers c
            WHERE c.follow_up_status = 'pending'
                AND c.lead_category IN ('HOT', 'WARM')
            ORDER BY c.lead_score DESC, c.last_interaction ASC
            LIMIT 10
            """,
            fetch_all=True
        )
        
        recommendations = []
        for row in (rows or []):
            d = dict(row)
            if d["lead_category"] == "HOT":
                d["action"] = "Call immediately — high-value prospect ready to buy"
                d["priority"] = "🔴 Urgent"
            else:
                d["action"] = "Follow up within 48 hours with product demo invitation"
                d["priority"] = "🟡 High"
            recommendations.append(d)
        
        return recommendations
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        return []

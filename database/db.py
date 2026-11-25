"""
Database connection manager for NeonDB (PostgreSQL).
Provides both sync (psycopg2) and async (asyncpg) connection pools.
"""

import os
import logging
from typing import Optional
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from database.schema import ALL_SCHEMA_SQL, CREATE_VECTOR_INDEX

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Global connection pool
_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None or _pool.closed:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is not set.")
        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=DATABASE_URL,
                cursor_factory=RealDictCursor
            )
            logger.info("Database connection pool created successfully.")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    return _pool


@contextmanager
def get_db():
    """Context manager for database connections."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        pool.putconn(conn)


def execute_query(sql: str, params=None, fetch_one=False, fetch_all=False):
    """Execute a SQL query with error handling."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch_one:
                return cur.fetchone()
            if fetch_all:
                return cur.fetchall()
            return None


def execute_many(sql: str, params_list: list):
    """Execute a SQL query with multiple parameter sets."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params_list)


def init_database():
    """Initialize database schema and seed data."""
    logger.info("Initializing database schema...")
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                # Create all tables
                for sql in ALL_SCHEMA_SQL:
                    cur.execute(sql)
                
                # Try vector index (may fail if not enough data yet)
                try:
                    cur.execute(CREATE_VECTOR_INDEX)
                except Exception:
                    pass  # Will be created after data is inserted
                
                conn.commit()
                logger.info("Database schema initialized successfully.")
        
        # Seed available appointment slots
        _seed_appointment_slots()
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise


def _seed_appointment_slots():
    """Generate available appointment slots for the next 30 days."""
    from datetime import date, time
    
    slots = []
    base_date = date.today()
    
    for day_offset in range(1, 31):  # Next 30 days
        slot_date = base_date + timedelta(days=day_offset)
        
        # Skip weekends
        if slot_date.weekday() >= 5:
            continue
        
        # Morning slots: 9 AM to 12 PM
        for hour in range(9, 12):
            slots.append((slot_date, time(hour, 0)))
        
        # Afternoon slots: 2 PM to 6 PM
        for hour in range(14, 18):
            slots.append((slot_date, time(hour, 0)))
    
    if slots:
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO available_slots (slot_date, slot_time, is_available)
                        VALUES (%s, %s, TRUE)
                        ON CONFLICT (slot_date, slot_time) DO NOTHING
                        """,
                        slots
                    )
                    conn.commit()
            logger.info(f"Seeded {len(slots)} appointment slots.")
        except Exception as e:
            logger.warning(f"Could not seed appointment slots: {e}")


def check_db_health() -> dict:
    """Check database connectivity and return health status."""
    try:
        result = execute_query("SELECT NOW() as current_time", fetch_one=True)
        return {
            "status": "healthy",
            "connected": True,
            "timestamp": str(result["current_time"]) if result else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }


def close_pool():
    """Close the connection pool gracefully."""
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        logger.info("Database connection pool closed.")

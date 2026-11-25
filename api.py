"""
FastAPI Backend Server for the Autonomous AI Sales Voice Agent.
Serves REST endpoints for the React frontend (default port 8000).
"""

import os
import io
import uuid
import base64
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import av
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def decode_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode browser audio (WebM/Opus/OGG/WAV) to mono float32 16kHz using PyAV."""
    container = av.open(io.BytesIO(raw_bytes))
    resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()
            chunks.append(arr.reshape(-1))
    # Flush resampler
    for resampled in resampler.resample(None):
        arr = resampled.to_ndarray()
        chunks.append(arr.reshape(-1))
    container.close()

    if not chunks:
        return np.array([], dtype=np.float32), 16000
    return np.concatenate(chunks).astype(np.float32), 16000


def _auto_ingest_sample_docs():
    """Auto-ingest sample documents if knowledge base is empty."""
    try:
        from rag.retriever import get_indexed_documents, generate_embedding, store_chunks, store_document
        from rag.ingest import ingest_document

        docs = get_indexed_documents()
        if len(docs) == 0:
            docs_dir = Path("rag/documents")
            if docs_dir.exists():
                for doc_path in sorted(docs_dir.glob("*.txt")):
                    try:
                        ingest_document(
                            str(doc_path),
                            embedding_fn=generate_embedding,
                            store_chunks_fn=store_chunks,
                            store_document_fn=store_document,
                        )
                    except Exception as e:
                        logger.warning(f"Could not ingest {doc_path.name}: {e}")
    except Exception as e:
        logger.warning(f"Auto-ingest skipped: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB schema and sample docs on server startup."""
    try:
        from database.db import init_database
        init_database()
        _auto_ingest_sample_docs()
        logger.info("FastAPI startup initialization complete.")
    except Exception as e:
        logger.error(f"Startup init warning: {e}")
    yield


app = FastAPI(
    title="TechVex AI Sales Voice Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REQUEST MODELS ───────────────────────────────────────────────────────────

class TextChatRequest(BaseModel):
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    message: str
    conversation_history: List[dict] = []
    include_tts: bool = True


class SummarizeRequest(BaseModel):
    session_id: str
    customer_id: Optional[str] = None
    conversation_history: List[dict] = []
    customer_context: dict = {}
    lead_score: float = 0.0
    lead_category: str = "COLD"
    sentiment: dict = {}
    booking_request: dict = {}


class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 4


class LoginRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


# ─── AUTH / CUSTOMER LOGIN ────────────────────────────────────────────────────

@app.post("/api/auth/login")
async def login_customer(req: LoginRequest):
    """Sign in or register a customer using name and/or email."""
    from services.crm import get_customer_by_email, get_customer_by_name, create_customer, update_customer

    def _do_login():
        name = (req.name or "").strip() or None
        email = (req.email or "").strip() or None
        if not name and not email:
            raise HTTPException(status_code=400, detail="Please provide at least your name or email.")

        existing = get_customer_by_email(email) if email else None
        if not existing and name:
            existing = get_customer_by_name(name)

        if existing:
            cid = str(existing["id"])
            updates = {}
            if name and not existing.get("name"):
                updates["name"] = name
            if email and not existing.get("email"):
                updates["email"] = email
            if updates:
                update_customer(cid, updates)
            display_name = name or existing.get("name") or (email.split("@")[0] if email else "there")
            display_email = email or existing.get("email") or ""
            return {
                "customer_id": cid,
                "name": display_name,
                "email": display_email,
                "returning": True,
                "greeting": f"Welcome back, {display_name}! How can I help you today?",
            }

        display_name = name or (email.split("@")[0] if email else "Guest")
        cid = create_customer({"name": display_name, "email": email})
        if not cid:
            raise HTTPException(status_code=500, detail="Could not create customer profile.")
        return {
            "customer_id": cid,
            "name": display_name,
            "email": email or "",
            "returning": False,
            "greeting": f"Hello {display_name}! Welcome to TechVex Solutions. How can I help you today?",
        }

    return await asyncio.to_thread(_do_login)


# ─── HEALTH ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def get_health():
    """Return health status concurrently for Database, LLM, STT, TTS, and RAG."""
    from database.db import check_db_health
    from agent.tools import check_ollama_health
    from voice.tts import check_tts_health
    from rag.retriever import check_rag_health

    db_res, llm_res, tts_res, rag_res = await asyncio.gather(
        asyncio.to_thread(check_db_health),
        asyncio.to_thread(check_ollama_health),
        asyncio.to_thread(check_tts_health),
        asyncio.to_thread(check_rag_health),
    )
    return {
        "database": db_res,
        "ollama": llm_res,
        "stt": {"status": "healthy", "model": os.getenv("WHISPER_MODEL", "small")},
        "tts": tts_res,
        "rag": rag_res,
    }


# ─── CHAT & VOICE ─────────────────────────────────────────────────────────────

@app.post("/api/chat/text")
async def chat_text(req: TextChatRequest):
    """Process a text message through the LangGraph sales agent without blocking the event loop."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    from agent.graph import process_message
    from voice.tts import synthesize_speech, audio_to_bytes

    session_id = req.session_id or str(uuid.uuid4())
    result = await asyncio.to_thread(
        process_message,
        session_id=session_id,
        message=req.message.strip(),
        customer_id=req.customer_id,
        conversation_history=req.conversation_history,
    )

    audio_b64 = None
    if req.include_tts and result.get("response"):
        try:
            tts_res = await asyncio.to_thread(
                synthesize_speech, result["response"], result.get("detected_language", "en")
            )
            if tts_res:
                arr, sr = tts_res
                wav_bytes = audio_to_bytes(arr, sr)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        except Exception as e:
            logger.warning(f"TTS skipped: {e}")

    result["audio_base64"] = audio_b64
    return result


@app.post("/api/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    customer_id: Optional[str] = Form(None),
    conversation_history: str = Form("[]"),
):
    """Process uploaded browser audio asynchronously: Decode -> STT -> LangGraph -> TTS."""
    import json
    from voice.stt import transcribe_audio
    from voice.tts import synthesize_speech, audio_to_bytes
    from agent.graph import process_message

    raw_bytes = await audio.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        audio_arr, sr = await asyncio.to_thread(decode_audio_bytes, raw_bytes)
    except Exception as e:
        logger.error(f"Audio decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    if len(audio_arr) < sr * 0.25:
        return {"empty_audio": True, "transcript": "", "response": ""}

    transcript, detected_lang, confidence = await asyncio.to_thread(transcribe_audio, audio_arr, sr)
    if not transcript or len(transcript.strip()) < 2:
        return {"empty_audio": True, "transcript": "", "response": ""}

    try:
        history = json.loads(conversation_history) if conversation_history else []
    except Exception:
        history = []

    sid = session_id or str(uuid.uuid4())
    result = await asyncio.to_thread(
        process_message,
        session_id=sid,
        message=transcript,
        customer_id=customer_id or None,
        conversation_history=history,
    )

    audio_b64 = None
    if result.get("response"):
        try:
            tts_res = await asyncio.to_thread(synthesize_speech, result["response"], detected_lang)
            if tts_res:
                arr, out_sr = tts_res
                wav_bytes = audio_to_bytes(arr, out_sr)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        except Exception as e:
            logger.warning(f"TTS error: {e}")

    result["transcript"] = transcript
    result["stt_confidence"] = round(confidence, 2)
    result["audio_base64"] = audio_b64
    result["empty_audio"] = False
    return result


@app.post("/api/chat/summarize")
async def summarize_call(req: SummarizeRequest):
    """Generate and persist structured call summary asynchronously."""
    from intelligence.summarizer import generate_call_summary, save_call_summary
    from services.crm import save_conversation, end_conversation
    from agent.tools import ollama_generate

    appointment = req.booking_request.get("appointment") if req.booking_request else None

    summary = await asyncio.to_thread(
        generate_call_summary,
        conversation_history=req.conversation_history,
        customer_context=req.customer_context,
        lead_score=req.lead_score,
        lead_category=req.lead_category,
        sentiment=req.sentiment,
        appointment=appointment,
        llm_fn=ollama_generate,
    )

    def _persist_summary():
        conv_id = save_conversation(req.session_id, req.customer_id)
        if conv_id and summary:
            save_call_summary(conv_id, req.customer_id, summary)
            end_conversation(conv_id, req.lead_score, req.sentiment.get("overall", "neutral"))

    await asyncio.to_thread(_persist_summary)
    return {"status": "ok", "summary": summary}


# ─── CRM ──────────────────────────────────────────────────────────────────────

@app.get("/api/customers")
async def list_customers(
    search: str = Query("", description="Search name, email, or company"),
    category: Optional[str] = Query(None, description="HOT, WARM, COLD"),
    sentiment: Optional[str] = Query(None, description="positive, neutral, negative"),
):
    """List and filter customers from NeonDB asynchronously."""
    from services.crm import search_customers

    cat = category if category and category != "All" else None
    sent = sentiment if sentiment and sentiment != "All" else None
    customers = await asyncio.to_thread(
        search_customers, query=search, lead_category=cat, sentiment=sent, limit=100
    )

    # Serialize UUID/Decimal/datetime cleanly
    cleaned = []
    for c in customers:
        cleaned.append({
            "id": str(c.get("id")),
            "name": c.get("name") or "Unknown",
            "email": c.get("email") or "",
            "phone": c.get("phone") or "",
            "company": c.get("company") or "",
            "company_size": c.get("company_size"),
            "budget": float(c["budget"]) if c.get("budget") is not None else None,
            "budget_currency": c.get("budget_currency") or "INR",
            "requirements": c.get("requirements") or "",
            "lead_score": float(c.get("lead_score") or 0),
            "lead_category": c.get("lead_category") or "COLD",
            "sentiment": c.get("sentiment") or "neutral",
            "preferred_language": c.get("preferred_language") or "en",
            "purchase_timeline": c.get("purchase_timeline") or "",
            "follow_up_status": c.get("follow_up_status") or "pending",
            "last_interaction": str(c.get("last_interaction"))[:19] if c.get("last_interaction") else "",
        })
    return {"customers": cleaned}


# ─── KNOWLEDGE BASE (RAG) ─────────────────────────────────────────────────────

@app.get("/api/documents")
async def list_documents():
    """List indexed documents asynchronously."""
    from rag.retriever import get_indexed_documents

    docs = await asyncio.to_thread(get_indexed_documents)
    return {
        "documents": [
            {
                "id": str(d.get("id")),
                "filename": d.get("filename"),
                "file_type": d.get("file_type"),
                "total_chunks": d.get("total_chunks", 0),
                "status": d.get("status"),
                "uploaded_at": str(d.get("uploaded_at"))[:19] if d.get("uploaded_at") else "",
            }
            for d in docs
        ]
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF, TXT, DOCX, or MD file into PgVector."""
    allowed = {".pdf", ".txt", ".docx", ".doc", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    from rag.ingest import ingest_document
    from rag.retriever import generate_embedding, store_chunks, store_document

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Preserve original filename by renaming or passing a temp copy in a temp dir
        tmp_dir = tempfile.mkdtemp()
        named_path = Path(tmp_dir) / Path(file.filename).name
        named_path.write_bytes(content)

        result = await asyncio.to_thread(
            ingest_document,
            str(named_path),
            embedding_fn=generate_embedding,
            store_chunks_fn=store_chunks,
            store_document_fn=store_document,
        )
        return result
    finally:
        try:
            os.unlink(tmp_path)
            if "named_path" in locals() and named_path.exists():
                named_path.unlink()
                Path(tmp_dir).rmdir()
        except Exception:
            pass


@app.post("/api/documents/query")
async def query_documents(req: RagQueryRequest):
    """Run semantic similarity search against PgVector asynchronously."""
    from rag.retriever import retrieve_relevant_chunks

    chunks = await asyncio.to_thread(retrieve_relevant_chunks, req.query, top_k=req.top_k)
    return {"chunks": chunks}


# ─── APPOINTMENTS ─────────────────────────────────────────────────────────────

@app.get("/api/appointments")
async def list_appointments():
    """List booked and cancelled appointments asynchronously."""
    from services.calendar import get_appointments

    appts = await asyncio.to_thread(get_appointments, limit=50)
    return {
        "appointments": [
            {
                "id": str(a.get("id")),
                "customer_name": a.get("customer_name") or "Unknown",
                "company": a.get("company") or "",
                "appointment_date": str(a.get("appointment_date")),
                "appointment_time": str(a.get("appointment_time"))[:5],
                "purpose": a.get("purpose") or "Product Demonstration",
                "status": a.get("status") or "confirmed",
            }
            for a in appts
        ]
    }


@app.get("/api/appointments/slots")
async def list_available_slots():
    """List upcoming available appointment slots asynchronously."""
    from services.calendar import get_available_slots

    return {"slots": await asyncio.to_thread(get_available_slots, days_ahead=14, limit=24)}


@app.post("/api/appointments/{appointment_id}/cancel")
async def cancel_appt(appointment_id: str):
    """Cancel an appointment and free its slot asynchronously."""
    from services.calendar import cancel_appointment

    ok = await asyncio.to_thread(cancel_appointment, appointment_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {"status": "cancelled", "id": appointment_id}


# ─── ANALYTICS ────────────────────────────────────────────────────────────────

@app.get("/api/analytics")
async def get_analytics():
    """Return KPIs, chart distributions, recent call summaries, and follow-up recommendations concurrently."""
    from intelligence.insights import (
        get_dashboard_kpis,
        get_lead_distribution_data,
        get_sentiment_data,
        get_calls_over_time,
        get_recent_summaries,
        get_follow_up_recommendations,
    )

    kpis, lead_dist, sent_dist, calls_time, summaries, recs = await asyncio.gather(
        asyncio.to_thread(get_dashboard_kpis),
        asyncio.to_thread(get_lead_distribution_data),
        asyncio.to_thread(get_sentiment_data),
        asyncio.to_thread(get_calls_over_time),
        asyncio.to_thread(get_recent_summaries, limit=10),
        asyncio.to_thread(get_follow_up_recommendations),
    )

    return {
        "kpis": kpis,
        "lead_distribution": lead_dist,
        "sentiment_distribution": sent_dist,
        "calls_over_time": calls_time,
        "recent_summaries": [
            {
                "id": str(s.get("id")),
                "customer_name": s.get("customer_name") or "Unknown",
                "company": s.get("company") or "",
                "lead_score": float(s.get("lead_score") or 0),
                "lead_category": s.get("lead_category") or "COLD",
                "sentiment": s.get("sentiment") or "neutral",
                "recommended_action": s.get("recommended_action") or "",
                "full_summary": s.get("full_summary") or "",
                "created_at": str(s.get("created_at"))[:16] if s.get("created_at") else "",
            }
            for s in summaries
        ],
        "recommendations": [
            {
                "id": str(r.get("id")),
                "name": r.get("name") or "Unknown",
                "company": r.get("company") or "",
                "lead_score": float(r.get("lead_score") or 0),
                "lead_category": r.get("lead_category") or "WARM",
                "action": r.get("action") or "",
                "priority": "Urgent" if r.get("lead_category") == "HOT" else "High",
            }
            for r in recs
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

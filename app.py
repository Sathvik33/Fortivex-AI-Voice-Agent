import os
import io
import uuid
import logging
import time
import json
import threading
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import gradio as gr
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── STARTUP HEALTH CHECKS ───────────────────────────────────────────────────

def startup_checks() -> dict:
    """Run all system health checks on startup."""
    results = {}
    
    # Database
    try:
        from database.db import init_database, check_db_health
        init_database()
        results["database"] = check_db_health()
        logger.info(f"Database: {results['database']['status']}")
    except Exception as e:
        results["database"] = {"status": "unhealthy", "error": str(e)}
        logger.error(f"Database init failed: {e}")
    
    # Ollama
    try:
        from agent.tools import check_ollama_health
        results["ollama"] = check_ollama_health()
        logger.info(f"Ollama: {results['ollama']['status']}")
    except Exception as e:
        results["ollama"] = {"status": "unhealthy", "error": str(e)}
    
    # STT (lazy load)
    results["stt"] = {"status": "ready", "model": os.getenv("WHISPER_MODEL", "small")}
    
    # TTS
    try:
        from voice.tts import check_tts_health
        results["tts"] = check_tts_health()
        logger.info(f"TTS: {results['tts']['status']}")
    except Exception as e:
        results["tts"] = {"status": "unhealthy", "error": str(e)}
    
    # RAG
    try:
        from rag.retriever import check_rag_health
        results["rag"] = check_rag_health()
        logger.info(f"RAG: {results['rag']['status']}")
    except Exception as e:
        results["rag"] = {"status": "unhealthy", "error": str(e)}
    
    # Pre-ingest sample documents if no documents indexed
    _auto_ingest_sample_docs()
    
    return results


def _auto_ingest_sample_docs():
    """Auto-ingest sample documents if knowledge base is empty."""
    try:
        from rag.retriever import get_indexed_documents, generate_embedding, store_chunks, store_document
        from rag.ingest import ingest_document
        
        docs = get_indexed_documents()
        if len(docs) == 0:
            logger.info("Auto-ingesting sample documents...")
            docs_dir = Path("rag/documents")
            if docs_dir.exists():
                for doc_path in sorted(docs_dir.glob("*.txt")):
                    try:
                        result = ingest_document(
                            str(doc_path),
                            embedding_fn=generate_embedding,
                            store_chunks_fn=store_chunks,
                            store_document_fn=store_document
                        )
                        logger.info(f"Auto-ingested: {result}")
                    except Exception as e:
                        logger.warning(f"Could not ingest {doc_path.name}: {e}")
    except Exception as e:
        logger.warning(f"Auto-ingest failed: {e}")


# ─── SESSION STATE ────────────────────────────────────────────────────────────

class AppState:
    """Global application state (thread-safe)."""
    _session_data: dict = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_session(cls, session_id: str) -> dict:
        with cls._lock:
            if session_id not in cls._session_data:
                cls._session_data[session_id] = {
                    "conversation_history": [],
                    "customer_id": None,
                    "lead_score": 0,
                    "lead_category": "COLD",
                    "sentiment": {"overall": "neutral"},
                    "customer_context": {},
                    "booking_request": {},
                    "is_active": False,
                    "total_messages": 0,
                }
            return cls._session_data[session_id]
    
    @classmethod
    def update_session(cls, session_id: str, data: dict):
        with cls._lock:
            if session_id in cls._session_data:
                cls._session_data[session_id].update(data)


# ─── VOICE PROCESSING ─────────────────────────────────────────────────────────

def process_voice_input(audio_input, session_id: str, session_state: dict):
    """Process audio input through the full pipeline."""
    if audio_input is None:
        return None, "", "", session_state
    
    try:
        from voice.stt import transcribe_audio
        from voice.tts import synthesize_speech
        from voice.realtime import numpy_to_gradio_audio
        from agent.graph import process_message
        
        # Parse audio
        if isinstance(audio_input, tuple):
            sr, audio = audio_input
        else:
            return None, "❌ Invalid audio format", "", session_state
        
        if audio is None or len(audio) == 0:
            return None, "🔇 No audio detected", "", session_state
        
        # Convert to float32
        if audio.dtype == np.int16:
            audio_f = audio.astype(np.float32) / 32768.0
        else:
            audio_f = audio.astype(np.float32)
        
        if len(audio_f.shape) > 1:
            audio_f = audio_f.mean(axis=1)
        
        # STT
        transcript, detected_lang, confidence = transcribe_audio(audio_f, sr)
        
        if not transcript or len(transcript.strip()) < 2:
            return None, "🎤 Could not understand audio. Please try again.", "", session_state
        
        # LLM via agent graph
        result = process_message(
            session_id=session_id,
            message=transcript,
            customer_id=session_state.get("customer_id"),
            conversation_history=session_state.get("conversation_history", [])
        )
        
        response_text = result.get("response", "I apologize, could you repeat that?")
        
        # Update session state
        session_state.update({
            "customer_id": result.get("customer_id", session_state.get("customer_id")),
            "conversation_history": result.get("conversation_history", []),
            "lead_score": result.get("lead_score", 0),
            "lead_category": result.get("lead_category", "COLD"),
            "sentiment": result.get("sentiment", {"overall": "neutral"}),
            "customer_context": result.get("customer_context", {}),
            "booking_request": result.get("booking_request", {}),
            "total_messages": session_state.get("total_messages", 0) + 1,
        })
        
        # TTS
        tts_result = synthesize_speech(response_text, detected_lang)
        
        audio_output = None
        if tts_result:
            audio_out, sr_out = tts_result
            audio_output = numpy_to_gradio_audio(audio_out, sr_out)
        
        return audio_output, f"🎤 You: {transcript}", f"🤖 Agent: {response_text}", session_state
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {str(e)[:100]}", "I'm having a technical issue. Please try again.", session_state


def process_text_input(text_input: str, session_id: str, session_state: dict):
    """Process text input through the agent (without voice)."""
    if not text_input or not text_input.strip():
        return "", session_state, gr.update(), gr.update()
    
    try:
        from agent.graph import process_message
        
        result = process_message(
            session_id=session_id,
            message=text_input.strip(),
            customer_id=session_state.get("customer_id"),
            conversation_history=session_state.get("conversation_history", [])
        )
        
        response_text = result.get("response", "I apologize, could you repeat that?")
        
        session_state.update({
            "customer_id": result.get("customer_id", session_state.get("customer_id")),
            "conversation_history": result.get("conversation_history", []),
            "lead_score": result.get("lead_score", 0),
            "lead_category": result.get("lead_category", "COLD"),
            "sentiment": result.get("sentiment", {"overall": "neutral"}),
            "customer_context": result.get("customer_context", {}),
            "booking_request": result.get("booking_request", {}),
            "total_messages": session_state.get("total_messages", 0) + 1,
        })
        
        # Format chat history for Gradio 6 Chatbot (list of {role, content} dicts)
        history = session_state.get("conversation_history", [])
        chat_messages = _format_chatbot_history(history)
        
        # Update context display
        ctx = session_state.get("customer_context", {}) or {}
        context_text = _format_context(ctx, session_state)
        
        return "", session_state, chat_messages, context_text
        
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        history = session_state.get("conversation_history", [])
        return "", session_state, [], f"❌ Error: {str(e)}"


def _format_chatbot_history(history: list) -> list[dict]:
    """Convert conversation history into Gradio 6 Chatbot message dicts."""
    messages = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            messages.append({"role": item["role"], "content": item["content"]})
    return messages


def _format_context(ctx: dict, session: dict) -> str:
    """Format customer context for display."""
    score = session.get("lead_score", 0)
    category = session.get("lead_category", "COLD")
    sentiment = session.get("sentiment", {})
    
    emoji_map = {"HOT": "🔥", "WARM": "🌡️", "COLD": "❄️"}
    sent_emoji = {"positive": "😊", "neutral": "😐", "negative": "😟"}
    
    lines = [
        f"**Lead Score:** {score:.0f}/100 {emoji_map.get(category, '')} {category}",
        f"**Sentiment:** {sent_emoji.get(sentiment.get('overall','neutral'), '')} {sentiment.get('overall', 'neutral')}",
        "---",
    ]
    
    if ctx.get("name"):
        lines.append(f"**Name:** {ctx['name']}")
    if ctx.get("company"):
        lines.append(f"**Company:** {ctx['company']}")
    if ctx.get("company_size"):
        lines.append(f"**Size:** {ctx['company_size']} employees")
    if ctx.get("budget"):
        currency = ctx.get("budget_currency", "INR")
        lines.append(f"**Budget:** {ctx['budget']:,.0f} {currency}")
    if ctx.get("purchase_timeline"):
        lines.append(f"**Timeline:** {ctx['purchase_timeline']}")
    if ctx.get("requirements"):
        req = str(ctx['requirements'])[:150]
        lines.append(f"**Requirements:** {req}...")
    
    booking = session.get("booking_request", {}) or {}
    if booking.get("appointment"):
        appt = booking["appointment"]
        lines.append(f"**📅 Appointment:** {appt.get('datetime_display', 'Confirmed')}")
    
    return "\n".join(lines) if len(lines) > 3 else "Start a conversation to see customer context here."


def end_call_and_summarize(session_id: str, session_state: dict):
    """End the current call and generate summary."""
    try:
        from intelligence.summarizer import generate_call_summary, save_call_summary
        from services.crm import end_conversation
        from agent.tools import ollama_generate
        
        history = session_state.get("conversation_history", [])
        ctx = session_state.get("customer_context", {}) or {}
        score = session_state.get("lead_score", 0)
        category = session_state.get("lead_category", "COLD")
        sentiment = session_state.get("sentiment", {}) or {}
        booking = session_state.get("booking_request", {}) or {}
        appointment = booking.get("appointment")
        
        summary = generate_call_summary(
            conversation_history=history,
            customer_context=ctx,
            lead_score=score,
            lead_category=category,
            sentiment=sentiment,
            appointment=appointment,
            llm_fn=ollama_generate
        )
        
        # Save to DB
        conv_id = session_state.get("crm_result", {}).get("conversation_id") if session_state.get("crm_result") else None
        if conv_id:
            save_call_summary(conv_id, session_state.get("customer_id"), summary)
            end_conversation(conv_id, score, sentiment.get("overall", "neutral"))
        
        # Format summary for display
        summary_text = f"""## 📋 Call Summary

**Customer:** {summary.get('customer_name', 'Unknown')}
**Company:** {summary.get('company', 'Unknown')}
**Lead Score:** {summary.get('lead_score', 0):.0f}/100 ({summary.get('lead_category', 'COLD')})
**Sentiment:** {summary.get('sentiment_summary', 'neutral')}
**Appointment:** {summary.get('appointment_details', 'None')}

**Requirements:** {summary.get('requirements', 'Not specified')}
**Budget:** {summary.get('budget', 'Not specified')}
**Timeline:** {summary.get('purchase_timeline', 'Not specified')}
**Objections:** {summary.get('objections', 'None')}

**🎯 Recommended Action:** {summary.get('recommended_action', 'Follow up')}

---
{summary.get('full_summary', '')}
"""
        return summary_text
        
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return f"❌ Could not generate summary: {str(e)}"


# ─── RAG FUNCTIONS ────────────────────────────────────────────────────────────

def upload_document(file_obj):
    """Handle document upload and ingestion."""
    if file_obj is None:
        return "❌ No file provided", get_documents_table()
    
    try:
        from rag.ingest import ingest_document
        from rag.retriever import generate_embedding, store_chunks, store_document
        
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        
        result = ingest_document(
            file_path=file_path,
            embedding_fn=generate_embedding,
            store_chunks_fn=store_chunks,
            store_document_fn=store_document
        )
        
        if result["status"] == "success":
            status = f"✅ Ingested '{result['filename']}' — {result['chunks']} chunks in {result['processing_time']}"
        elif result["status"] == "duplicate":
            status = f"ℹ️ '{result['filename']}' already indexed"
        else:
            status = f"❌ Failed: {result.get('error', 'Unknown error')}"
        
        return status, get_documents_table()
        
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        return f"❌ Upload error: {str(e)}", get_documents_table()


def get_documents_table():
    """Get indexed documents as DataFrame."""
    try:
        import pandas as pd
        from rag.retriever import get_indexed_documents
        
        docs = get_indexed_documents()
        if not docs:
            return pd.DataFrame(columns=["Filename", "Type", "Chunks", "Status", "Uploaded"])
        
        rows = []
        for doc in docs:
            rows.append({
                "Filename": doc.get("filename", ""),
                "Type": doc.get("file_type", ""),
                "Chunks": doc.get("total_chunks", 0),
                "Status": "✅ Ready" if doc.get("status") == "ready" else "⏳ " + doc.get("status", ""),
                "Uploaded": str(doc.get("uploaded_at", ""))[:16]
            })
        
        return pd.DataFrame(rows)
    except Exception as e:
        import pandas as pd
        logger.error(f"Document table error: {e}")
        return pd.DataFrame()


def test_rag_query(query: str):
    """Test RAG retrieval with a query."""
    if not query:
        return "Enter a query to test RAG retrieval."
    
    try:
        from rag.retriever import retrieve_relevant_chunks, format_rag_context
        
        chunks = retrieve_relevant_chunks(query, top_k=3)
        
        if not chunks:
            return "No relevant documents found. Make sure documents are indexed."
        
        result_parts = [f"**Found {len(chunks)} relevant chunk(s):**\n"]
        for i, chunk in enumerate(chunks, 1):
            result_parts.append(
                f"**[{i}] Source:** `{chunk['source']}` (similarity: {chunk['similarity']:.0%})\n"
                f"{chunk['content'][:300]}...\n"
            )
        
        return "\n".join(result_parts)
    except Exception as e:
        return f"❌ RAG test error: {str(e)}"


# ─── CRM FUNCTIONS ────────────────────────────────────────────────────────────

def get_customers_table(search="", category_filter="All", sentiment_filter="All"):
    """Get customers as DataFrame."""
    import pandas as pd
    
    try:
        from services.crm import search_customers
        
        cat = category_filter if category_filter != "All" else None
        sent = sentiment_filter if sentiment_filter != "All" else None
        
        customers = search_customers(query=search, lead_category=cat, sentiment=sent, limit=100)
        
        if not customers:
            return pd.DataFrame(columns=["Name", "Company", "Email", "Score", "Category", "Sentiment", "Budget", "Last Contact"])
        
        rows = []
        for c in customers:
            budget = c.get("budget")
            budget_str = f"₹{float(budget):,.0f}" if budget else "—"
            
            cat_emoji = {"HOT": "🔥 HOT", "WARM": "🌡️ WARM", "COLD": "❄️ COLD"}.get(c.get("lead_category", "COLD"), "COLD")
            
            rows.append({
                "Name": c.get("name", "Unknown"),
                "Company": c.get("company", "—"),
                "Email": c.get("email", "—"),
                "Score": f"{float(c.get('lead_score', 0)):.0f}",
                "Category": cat_emoji,
                "Sentiment": c.get("sentiment", "neutral"),
                "Budget": budget_str,
                "Last Contact": str(c.get("last_interaction", ""))[:10] if c.get("last_interaction") else "—"
            })
        
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"CRM table error: {e}")
        return pd.DataFrame()


# ─── APPOINTMENT FUNCTIONS ───────────────────────────────────────────────────

def get_appointments_table():
    """Get appointments as DataFrame."""
    import pandas as pd
    
    try:
        from services.calendar import get_appointments
        
        appts = get_appointments(limit=50)
        
        if not appts:
            return pd.DataFrame(columns=["Customer", "Company", "Date", "Time", "Purpose", "Status"])
        
        rows = []
        for a in appts:
            status_emoji = {"confirmed": "✅", "cancelled": "❌", "pending": "⏳"}.get(a.get("status", ""), "❓")
            rows.append({
                "Customer": a.get("customer_name", "Unknown"),
                "Company": a.get("company", "—"),
                "Date": str(a.get("appointment_date", ""))[:10],
                "Time": str(a.get("appointment_time", ""))[:5],
                "Purpose": a.get("purpose", "Demo"),
                "Status": f"{status_emoji} {a.get('status', 'unknown').title()}"
            })
        
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Appointments table error: {e}")
        return pd.DataFrame()


def get_available_slots_table():
    """Get available appointment slots."""
    import pandas as pd
    
    try:
        from services.calendar import get_available_slots
        
        slots = get_available_slots(days_ahead=14, limit=20)
        
        if not slots:
            return pd.DataFrame(columns=["Date", "Day", "Time", "Availability"])
        
        rows = [{"Date": s["date"], "Day": s["day_name"], "Time": s["time"], "Availability": "✅ Available"} for s in slots]
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame()


def cancel_appointment_by_id(appointment_id: str):
    """Cancel an appointment."""
    if not appointment_id or not appointment_id.strip():
        return "❌ Please enter a valid Appointment ID", get_appointments_table()
    
    try:
        from services.calendar import cancel_appointment
        
        success = cancel_appointment(appointment_id.strip())
        
        if success:
            return f"✅ Appointment {appointment_id} cancelled successfully.", get_appointments_table()
        else:
            return f"❌ Could not find appointment {appointment_id}.", get_appointments_table()
    except Exception as e:
        return f"❌ Error: {str(e)}", get_appointments_table()


# ─── ANALYTICS FUNCTIONS ──────────────────────────────────────────────────────

def refresh_analytics():
    """Refresh all analytics data."""
    from intelligence.insights import (
        get_dashboard_kpis, get_lead_distribution_data,
        get_sentiment_data, get_calls_over_time, get_recent_summaries,
        get_follow_up_recommendations
    )
    from dashboard.analytics import (
        create_lead_distribution_chart, create_sentiment_chart,
        create_calls_timeline_chart, format_kpi_cards, format_recommendations_html
    )
    
    kpis = get_dashboard_kpis()
    lead_data = get_lead_distribution_data()
    sentiment_data = get_sentiment_data()
    calls_data = get_calls_over_time()
    summaries = get_recent_summaries(limit=5)
    recs = get_follow_up_recommendations()
    
    kpi_html = format_kpi_cards(kpis)
    lead_chart = create_lead_distribution_chart(lead_data)
    sentiment_chart = create_sentiment_chart(sentiment_data)
    calls_chart = create_calls_timeline_chart(calls_data)
    recs_html = format_recommendations_html(recs)
    
    # Recent summaries text
    if summaries:
        summary_parts = []
        for s in summaries:
            summary_parts.append(
                f"**{s.get('customer_name', 'Unknown')}** ({s.get('company', '—')}) — "
                f"Score: {s.get('lead_score', 0):.0f} | {s.get('lead_category', 'COLD')} | "
                f"{str(s.get('created_at', ''))[:10]}\n"
                f"> {s.get('recommended_action', 'Follow up')}"
            )
        summaries_text = "\n\n".join(summary_parts)
    else:
        summaries_text = "No call summaries yet. Complete a call to see summaries here."
    
    return kpi_html, lead_chart, sentiment_chart, calls_chart, summaries_text, recs_html


# ─── GRADIO UI ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* Global styles */
.gradio-container {
    background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 100%) !important;
    min-height: 100vh;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #6C63FF 0%, #4B9EFF 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(108, 99, 255, 0.4);
}

/* Status indicators */
.status-healthy { color: #00CC44; font-weight: bold; }
.status-unhealthy { color: #FF4B4B; font-weight: bold; }

/* Chat bubbles */
.message.user { background: #6C63FF !important; }
.message.bot { background: rgba(255,255,255,0.08) !important; }

/* Score badge */
.score-hot { background: #FF4B4B; color: white; padding: 4px 12px; border-radius: 20px; }
.score-warm { background: #FFA500; color: white; padding: 4px 12px; border-radius: 20px; }
.score-cold { background: #4B9EFF; color: white; padding: 4px 12px; border-radius: 20px; }

/* Tabs */
.tab-nav { background: rgba(255,255,255,0.05) !important; }
"""


def build_ui():
    """Build the full Gradio interface."""
    
    with gr.Blocks(title="🤖 TechVex AI Sales Agent") as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1 style="color:white; font-size:28px; margin:0; font-weight:800;">
                🤖 TechVex Autonomous AI Sales Agent
            </h1>
            <p style="color:rgba(255,255,255,0.8); margin:8px 0 0; font-size:15px;">
                Real-time Voice · Multilingual · LangGraph · RAG · CRM · Analytics
            </p>
        </div>
        """)
        
        # Session state
        session_id_state = gr.State(lambda: str(uuid.uuid4()))
        session_data_state = gr.State(lambda: AppState.get_session(str(uuid.uuid4())))
        
        with gr.Tabs() as tabs:
            
            # ═══════════════════════════════════════════════════════════════
            # TAB A: LIVE SALES AGENT
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🎙️ Live Sales Agent", id="voice_tab"):
                
                gr.HTML("""
                <div style="background:rgba(108,99,255,0.1); border:1px solid rgba(108,99,255,0.3); 
                            padding:12px 16px; border-radius:8px; margin-bottom:16px;">
                    <strong>🎤 Voice Mode:</strong> Record your message below → Agent processes speech → 
                    Responds in audio + text. Or type below for text-only mode.
                    <br><small style="color:orange;">⚠️ Note: Recording consent is required. By using this, you consent to AI-assisted sales interaction.</small>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Chat display
                        chatbot = gr.Chatbot(
                            label="💬 Conversation",
                            height=400,
                            show_label=True,
                            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=techvex"),
                        )
                        
                        # Voice input
                        with gr.Group():
                            gr.HTML("<p style='color:#aaa; font-size:13px; margin-bottom:8px;'>🎙️ <b>Voice Input</b> — Record, then click Send:</p>")
                            audio_input = gr.Audio(
                                sources=["microphone"],
                                type="numpy",
                                label="Speak here",
                                show_label=False,
                            )
                            audio_output = gr.Audio(
                                label="🔊 Agent Audio Response",
                                autoplay=True,
                                show_label=True,
                            )
                            voice_send_btn = gr.Button("🎙️ Process Voice", variant="primary", size="sm")
                        
                        # Text input (alternative)
                        with gr.Group():
                            gr.HTML("<p style='color:#aaa; font-size:13px; margin:8px 0;'>⌨️ <b>Or type your message:</b></p>")
                            with gr.Row():
                                text_input = gr.Textbox(
                                    placeholder="Type your message here...",
                                    show_label=False,
                                    scale=4,
                                )
                                text_send_btn = gr.Button("Send ➤", variant="primary", scale=1)
                        
                        # STT/Response display
                        transcript_display = gr.Textbox(
                            label="📝 Recognized Speech",
                            interactive=False,
                            show_label=True,
                            max_lines=2,
                        )
                        response_display = gr.Textbox(
                            label="💡 Agent Response Text",
                            interactive=False,
                            show_label=True,
                            max_lines=4,
                        )
                        
                        # Action buttons
                        with gr.Row():
                            end_call_btn = gr.Button("📋 End Call & Summarize", variant="secondary")
                            new_session_btn = gr.Button("🔄 New Session", variant="stop")
                    
                    with gr.Column(scale=1):
                        # Customer Context Panel
                        gr.HTML("<h3 style='color:#6C63FF; margin:0 0 12px;'>👤 Customer Context</h3>")
                        context_display = gr.Markdown(
                            "Start a conversation to see customer context here.",
                            label="Customer Context"
                        )
                        
                        gr.HTML("<h3 style='color:#6C63FF; margin:16px 0 12px;'>📋 Call Summary</h3>")
                        summary_display = gr.Markdown(
                            "End the call to generate a summary.",
                        )
                
                # Status bar
                with gr.Row():
                    status_text = gr.HTML("""
                    <div style="background:rgba(0,204,68,0.1); border:1px solid rgba(0,204,68,0.3); 
                                padding:8px 16px; border-radius:8px; color:#00CC44; font-size:13px;">
                        ✅ System Ready — Speak or type to begin
                    </div>
                    """)
                
                # Event handlers
                def handle_voice(audio, sid, sdata):
                    if not sdata:
                        sdata = AppState.get_session(sid)
                    audio_out, transcript, response, updated_sdata = process_voice_input(audio, sid, sdata)
                    
                    # Update chat
                    history = updated_sdata.get("conversation_history", [])
                    chat_hist = _format_chatbot_history(history)
                    
                    ctx_text = _format_context(updated_sdata.get("customer_context", {}), updated_sdata)
                    return audio_out, transcript, response, chat_hist, ctx_text, updated_sdata
                
                def handle_text(text, sid, sdata):
                    if not text.strip():
                        return "", sdata, [], ""
                    if not sdata:
                        sdata = AppState.get_session(sid)
                    text_out, updated_sdata, chat_hist, ctx_text = process_text_input(text, sid, sdata)
                    return text_out, updated_sdata, chat_hist, ctx_text
                
                def handle_end_call(sid, sdata):
                    if not sdata:
                        return "No active session to summarize."
                    return end_call_and_summarize(sid, sdata)
                
                def handle_new_session():
                    new_sid = str(uuid.uuid4())
                    new_sdata = AppState.get_session(new_sid)
                    return new_sid, new_sdata, [], "New session started.", "", "", None, None
                
                voice_send_btn.click(
                    handle_voice,
                    inputs=[audio_input, session_id_state, session_data_state],
                    outputs=[audio_output, transcript_display, response_display, chatbot, context_display, session_data_state]
                )
                
                text_send_btn.click(
                    handle_text,
                    inputs=[text_input, session_id_state, session_data_state],
                    outputs=[text_input, session_data_state, chatbot, context_display]
                )
                
                text_input.submit(
                    handle_text,
                    inputs=[text_input, session_id_state, session_data_state],
                    outputs=[text_input, session_data_state, chatbot, context_display]
                )
                
                end_call_btn.click(
                    handle_end_call,
                    inputs=[session_id_state, session_data_state],
                    outputs=[summary_display]
                )
                
                new_session_btn.click(
                    handle_new_session,
                    outputs=[session_id_state, session_data_state, chatbot, context_display, transcript_display, response_display, audio_input, audio_output]
                )
            
            # ═══════════════════════════════════════════════════════════════
            # TAB B: CUSTOMER CRM
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("👥 Customer CRM", id="crm_tab"):
                
                gr.HTML("<h3 style='color:#6C63FF;'>Customer Records</h3>")
                
                with gr.Row():
                    crm_search = gr.Textbox(placeholder="Search by name, email, company...", label="Search", scale=3)
                    crm_category = gr.Dropdown(["All", "HOT", "WARM", "COLD"], value="All", label="Lead Category")
                    crm_sentiment = gr.Dropdown(["All", "positive", "neutral", "negative"], value="All", label="Sentiment")
                    crm_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                
                crm_table = gr.DataFrame(
                    get_customers_table,
                    label="Customers",
                    interactive=False,
                    wrap=True,
                )
                
                crm_refresh_btn.click(
                    get_customers_table,
                    inputs=[crm_search, crm_category, crm_sentiment],
                    outputs=[crm_table]
                )
                crm_search.change(
                    get_customers_table,
                    inputs=[crm_search, crm_category, crm_sentiment],
                    outputs=[crm_table]
                )
            
            # ═══════════════════════════════════════════════════════════════
            # TAB C: KNOWLEDGE BASE
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📚 Knowledge Base", id="rag_tab"):
                
                gr.HTML("<h3 style='color:#6C63FF;'>Document Knowledge Base (RAG)</h3>")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h4>Upload Documents</h4>")
                        doc_upload = gr.File(
                            label="Upload PDF, TXT, or DOCX",
                            file_types=[".pdf", ".txt", ".docx", ".md"],
                        )
                        upload_btn = gr.Button("📤 Ingest Document", variant="primary")
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        
                        gr.HTML("<h4 style='margin-top:16px;'>Test RAG Retrieval</h4>")
                        rag_query = gr.Textbox(
                            placeholder="e.g., What are the pricing plans?",
                            label="Test Query"
                        )
                        rag_test_btn = gr.Button("🔍 Test Retrieval")
                        rag_result = gr.Markdown(label="Retrieved Context")
                    
                    with gr.Column(scale=2):
                        gr.HTML("<h4>Indexed Documents</h4>")
                        docs_table = gr.DataFrame(
                            get_documents_table,
                            label="Indexed Documents",
                            interactive=False,
                        )
                        docs_refresh_btn = gr.Button("🔄 Refresh Documents")
                
                upload_btn.click(
                    upload_document,
                    inputs=[doc_upload],
                    outputs=[upload_status, docs_table]
                )
                docs_refresh_btn.click(get_documents_table, outputs=[docs_table])
                rag_test_btn.click(test_rag_query, inputs=[rag_query], outputs=[rag_result])
            
            # ═══════════════════════════════════════════════════════════════
            # TAB D: APPOINTMENTS
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📅 Appointments", id="appointments_tab"):
                
                gr.HTML("<h3 style='color:#6C63FF;'>Appointment Management</h3>")
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h4>✅ Available Slots (Next 14 Days)</h4>")
                        slots_table = gr.DataFrame(
                            get_available_slots_table,
                            label="Available Slots",
                            interactive=False,
                        )
                        slots_refresh_btn = gr.Button("🔄 Refresh Slots")
                    
                    with gr.Column():
                        gr.HTML("<h4>📋 All Appointments</h4>")
                        appts_table = gr.DataFrame(
                            get_appointments_table,
                            label="Appointments",
                            interactive=False,
                        )
                        appts_refresh_btn = gr.Button("🔄 Refresh Appointments")
                        
                        gr.HTML("<h4 style='margin-top:16px;'>❌ Cancel Appointment</h4>")
                        cancel_id_input = gr.Textbox(placeholder="Paste Appointment ID here...", label="Appointment ID")
                        cancel_btn = gr.Button("Cancel Appointment", variant="stop")
                        cancel_status = gr.Textbox(label="Status", interactive=False)
                
                slots_refresh_btn.click(get_available_slots_table, outputs=[slots_table])
                appts_refresh_btn.click(get_appointments_table, outputs=[appts_table])
                cancel_btn.click(
                    cancel_appointment_by_id,
                    inputs=[cancel_id_input],
                    outputs=[cancel_status, appts_table]
                )
            
            # ═══════════════════════════════════════════════════════════════
            # TAB E: ANALYTICS DASHBOARD
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📊 Analytics", id="analytics_tab"):
                
                gr.HTML("<h3 style='color:#6C63FF;'>Business Intelligence Dashboard</h3>")
                
                analytics_refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
                
                # KPI Cards
                kpi_html_display = gr.HTML(label="KPIs")
                
                with gr.Row():
                    lead_chart = gr.Plot(label="Lead Distribution")
                    sentiment_chart_display = gr.Plot(label="Sentiment Distribution")
                
                calls_chart_display = gr.Plot(label="Call Volume Trend")
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h4>📋 Recent Call Summaries</h4>")
                        summaries_display = gr.Markdown("Click Refresh to load summaries.")
                    
                    with gr.Column():
                        gr.HTML("<h4>🎯 Follow-up Recommendations</h4>")
                        recommendations_display = gr.HTML("Click Refresh to load recommendations.")
                
                analytics_refresh_btn.click(
                    refresh_analytics,
                    outputs=[kpi_html_display, lead_chart, sentiment_chart_display, calls_chart_display, summaries_display, recommendations_display]
                )
                
                # Auto-load analytics on tab select
                app.load(
                    refresh_analytics,
                    outputs=[kpi_html_display, lead_chart, sentiment_chart_display, calls_chart_display, summaries_display, recommendations_display]
                )
        
        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:16px; color:rgba(255,255,255,0.4); font-size:12px; margin-top:16px;">
            🤖 TechVex AI Sales Agent · Built with LangGraph, Faster-Whisper, Piper TTS, Ollama (Qwen2.5-7B), PgVector
            <br>⚠️ Prototype — Uses locally stored customer data. Not production-ready without additional security hardening.
        </div>
        """)
    
    return app


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    logger.info("=" * 60)
    logger.info("  TECHVEX AUTONOMOUS AI SALES AGENT — STARTING UP")
    logger.info("=" * 60)
    
    # Run health checks
    logger.info("Running system health checks...")
    health = startup_checks()
    
    for component, status in health.items():
        status_str = status.get("status", "unknown")
        icon = "✅" if status_str == "healthy" else "⚠️"
        logger.info(f"  {icon} {component}: {status_str}")
    
    # Build and launch UI
    logger.info("Building Gradio interface...")
    app = build_ui()
    
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "7860"))
    
    logger.info(f"Launching on http://{host}:{port}")
    logger.info("=" * 60)
    
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True,
        inbrowser=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
        ),
    )

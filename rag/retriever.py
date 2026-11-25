"""
RAG module: Semantic retrieval from PgVector (NeonDB).
Embedding generation with sentence-transformers, similarity search.
"""

import os
import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))

# Singleton embedding model
_embedding_model = None
_embedding_lock = threading.Lock()


def load_embedding_model():
    """Load sentence-transformers embedding model (singleton)."""
    global _embedding_model
    
    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            start = time.time()
            
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            elapsed = time.time() - start
            logger.info(f"Embedding model loaded in {elapsed:.2f}s")
            return _embedding_model
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise


def generate_embedding(text: str) -> list[float]:
    """Generate embedding vector for text."""
    model = load_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts efficiently."""
    model = load_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def store_document(filename: str, file_type: str, content_hash: str, total_chunks: int) -> Optional[str]:
    """Store document metadata in database. Returns document ID or None if duplicate."""
    from database.db import execute_query
    
    try:
        # Check for duplicate
        existing = execute_query(
            "SELECT id FROM documents WHERE content_hash = %s",
            (content_hash,),
            fetch_one=True
        )
        
        if existing:
            logger.info(f"Document already exists: {filename}")
            return None
        
        result = execute_query(
            """
            INSERT INTO documents (filename, file_type, content_hash, total_chunks, status)
            VALUES (%s, %s, %s, %s, 'processing')
            RETURNING id
            """,
            (filename, file_type, content_hash, total_chunks),
            fetch_one=True
        )
        
        return str(result["id"]) if result else None
        
    except Exception as e:
        logger.error(f"Failed to store document: {e}")
        return None


def store_chunks(chunks: list[dict]):
    """Store document chunks with embeddings in PgVector."""
    from database.db import get_db
    from pgvector.psycopg2 import register_vector
    import psycopg2.extras
    
    if not chunks:
        return
    
    with get_db() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for chunk in chunks:
                cur.execute(
                    """
                    INSERT INTO document_chunks (document_id, chunk_index, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        chunk["document_id"],
                        chunk["chunk_index"],
                        chunk["content"],
                        chunk["embedding"],
                        psycopg2.extras.Json(chunk.get("metadata", {}))
                    )
                )
            
            # Mark document as processed
            if chunks:
                cur.execute(
                    "UPDATE documents SET status = 'ready', processed_at = NOW() WHERE id = %s",
                    (chunk["document_id"],)
                )
            
            conn.commit()
    
    logger.info(f"Stored {len(chunks)} chunks in PgVector")


def retrieve_relevant_chunks(
    query: str,
    top_k: int = TOP_K,
    min_similarity: float = 0.3
) -> list[dict]:
    """
    Retrieve most relevant document chunks using cosine similarity.
    
    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        min_similarity: Minimum cosine similarity threshold
    
    Returns:
        List of relevant chunk dicts with content and metadata
    """
    from database.db import get_db
    from pgvector.psycopg2 import register_vector
    
    try:
        query_embedding = generate_embedding(query)
        
        with get_db() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        dc.content,
                        dc.metadata,
                        d.filename,
                        1 - (dc.embedding <=> %s::vector) AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.status = 'ready'
                        AND 1 - (dc.embedding <=> %s::vector) > %s
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, min_similarity, query_embedding, top_k)
                )
                
                rows = cur.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "content": row["content"],
                "source": row["filename"],
                "similarity": float(row["similarity"]),
                "metadata": row["metadata"] or {}
            })
        
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}'")
        return results
        
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []


def format_rag_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into LLM context."""
    if not chunks:
        return ""
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        content = chunk.get("content", "")
        similarity = chunk.get("similarity", 0)
        context_parts.append(
            f"[Source {i}: {source} (relevance: {similarity:.0%})]\n{content}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def get_indexed_documents() -> list[dict]:
    """Get list of all indexed documents."""
    from database.db import execute_query
    
    try:
        rows = execute_query(
            """
            SELECT id, filename, file_type, total_chunks, status, uploaded_at, processed_at
            FROM documents
            ORDER BY uploaded_at DESC
            """,
            fetch_all=True
        )
        return [dict(row) for row in rows] if rows else []
    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        return []


def check_rag_health() -> dict:
    """Check RAG system health."""
    try:
        model = load_embedding_model()
        docs = get_indexed_documents()
        return {
            "status": "healthy",
            "embedding_model": EMBEDDING_MODEL,
            "indexed_documents": len(docs),
            "ready_documents": len([d for d in docs if d.get("status") == "ready"])
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

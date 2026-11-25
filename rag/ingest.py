"""
RAG module: Document ingestion pipeline.
Supports PDF, TXT, DOCX extraction → chunking → embedding → PgVector storage.
"""

import os
import io
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import pymupdf
        doc = pymupdf.open(file_path)
        texts = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                texts.append(text.strip())
        doc.close()
        return "\n\n".join(texts)
    except Exception as e:
        logger.error(f"PyMuPDF failed: {e}, trying pypdf")
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e2:
            logger.error(f"pypdf also failed: {e2}")
            raise


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """Extract text from any supported document type."""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    extractors = {
        '.pdf': extract_text_from_pdf,
        '.docx': extract_text_from_docx,
        '.doc': extract_text_from_docx,
        '.txt': extract_text_from_txt,
        '.md': extract_text_from_txt,
    }
    
    extractor = extractors.get(ext)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return extractor(file_path)


def clean_text(text: str) -> str:
    """Clean extracted text."""
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Hyphenated line breaks
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Tries to split at sentence boundaries.
    """
    if not text:
        return []
    
    # Split into sentences first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Keep overlap
            overlap_words = ' '.join(current_chunk).split()[-overlap:]
            current_chunk = overlap_words + [sentence]
            current_size = len(overlap_words) + sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [c for c in chunks if len(c.strip()) > 50]  # Filter tiny chunks


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file for deduplication."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def ingest_document(
    file_path: str,
    embedding_fn,
    store_chunks_fn,
    store_document_fn,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> dict:
    """
    Full document ingestion pipeline.
    
    Args:
        file_path: Path to document
        embedding_fn: Function to generate embeddings
        store_chunks_fn: Function to store chunks in DB
        store_document_fn: Function to store document metadata
        chunk_size: Words per chunk
        overlap: Overlap words between chunks
    
    Returns:
        dict with ingestion status
    """
    start_time = time.time()
    path = Path(file_path)
    
    logger.info(f"Ingesting document: {path.name}")
    
    # Compute hash for deduplication
    file_hash = compute_file_hash(file_path)
    
    # Extract text
    try:
        raw_text = extract_text(file_path)
        clean = clean_text(raw_text)
    except Exception as e:
        return {"status": "error", "error": f"Text extraction failed: {e}", "filename": path.name}
    
    if not clean:
        return {"status": "error", "error": "No text extracted", "filename": path.name}
    
    # Chunk text
    chunks = chunk_text(clean, chunk_size, overlap)
    
    if not chunks:
        return {"status": "error", "error": "No chunks generated", "filename": path.name}
    
    # Store document metadata
    doc_id = store_document_fn(
        filename=path.name,
        file_type=path.suffix.lower().lstrip('.'),
        content_hash=file_hash,
        total_chunks=len(chunks)
    )
    
    if doc_id is None:
        return {"status": "duplicate", "message": "Document already indexed", "filename": path.name}
    
    # Generate embeddings and store chunks
    embedded_chunks = []
    for i, chunk_text_content in enumerate(chunks):
        try:
            embedding = embedding_fn(chunk_text_content)
            embedded_chunks.append({
                "document_id": doc_id,
                "chunk_index": i,
                "content": chunk_text_content,
                "embedding": embedding,
                "metadata": {
                    "source": path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        except Exception as e:
            logger.warning(f"Embedding failed for chunk {i}: {e}")
    
    if embedded_chunks:
        store_chunks_fn(embedded_chunks)
    
    elapsed = time.time() - start_time
    
    result = {
        "status": "success",
        "filename": path.name,
        "chunks": len(chunks),
        "embedded_chunks": len(embedded_chunks),
        "processing_time": f"{elapsed:.2f}s",
        "doc_id": str(doc_id)
    }
    
    logger.info(f"Ingested {path.name}: {len(chunks)} chunks in {elapsed:.2f}s")
    return result

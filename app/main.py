from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.routes import router
from dotenv import load_dotenv
from services.ocr import extract_text_from_pdf
from services.chunking import bangla_exam_chunker
from services.embeddings import embed_query
from services.retrieval import retrieve_context_with_metadata
from app.models import DocumentChunk
import io
import sqlite3
import pickle
import shutil
import os

load_dotenv()

app = FastAPI(title="Bengali/English RAG Chatbot")
app.include_router(router)

DB_PATH = 'chunks.db'


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT,
        text TEXT,
        embedding BLOB,
        meta BLOB
    )''')
    conn.commit()
    conn.close()


def store_chunks_in_db(chunks):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for chunk in chunks:
        c.execute('''INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, embedding, meta) VALUES (?, ?, ?, ?, ?)''',
            (chunk.chunk_id, chunk.doc_id, chunk.text, pickle.dumps(chunk.embedding), pickle.dumps(chunk.meta.dict()))
        )
    conn.commit()
    conn.close()


def load_chunks_from_db(doc_id: str = None):
    # Ensure doc_id is a string
    if doc_id is None:
        doc_id = ''
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if doc_id:
        c.execute('SELECT chunk_id, doc_id, text, embedding, meta FROM chunks WHERE doc_id=?', (doc_id,))
    else:
        c.execute('SELECT chunk_id, doc_id, text, embedding, meta FROM chunks')
    rows = c.fetchall()
    conn.close()
    from app.models import DocumentChunk, ChunkMeta
    chunks = []
    for chunk_id, doc_id, text, embedding, meta in rows:
        meta_obj = ChunkMeta(**pickle.loads(meta))
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            embedding=pickle.loads(embedding),
            meta=meta_obj
        )
        chunks.append(chunk)
    return chunks


# End-to-end pipeline: PDF -> Chunks with embeddings


def process_bangla_exam_pdf(pdf_path: str, doc_id: str) -> list:
    with open(pdf_path, 'rb') as f:
        pdf_bytes = io.BytesIO(f.read())
    text = extract_text_from_pdf(pdf_bytes, language='ben+eng')
    chunks = bangla_exam_chunker(text, doc_id)
    # Embed each chunk
    for chunk in chunks:
        chunk.embedding = embed_query(chunk.text)
    return chunks  # List[DocumentChunk]


# End-to-end query: retrieve best context for a question


def answer_bangla_exam_query(query: str, chunks: list, section=None, question_number=None, chunk_type=None, top_k=3):
    # Retrieve top-k relevant chunks using metadata and semantic similarity
    results = retrieve_context_with_metadata(query, chunks, section, question_number, chunk_type, top_k=top_k)
    # Compose answer (for demo, just return the top chunk's text and context)
    if results:
        answer = results[0].text
        context = [c.text for c in results]
    else:
        answer = "No relevant answer found."
        context = []
    return {
        'answer': answer,
        'context': context
    }


@app.on_event("startup")
def startup_event():
    init_db()


@app.post("/upload_pdf/")
def upload_pdf(doc_id: str = Form(...), file: UploadFile = File(...)):
    # Save uploaded PDF
    pdf_path = f"uploaded_{doc_id}.pdf"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Ensure doc_id is a string
    safe_doc_id = doc_id if doc_id is not None else ''
    # Process and store chunks
    chunks = process_bangla_exam_pdf(pdf_path, safe_doc_id)
    store_chunks_in_db(chunks)
    os.remove(pdf_path)
    return {"message": f"Document {doc_id} processed and stored.", "chunk_count": len(chunks)}


@app.get("/qa/")
def qa(query: str, doc_id: str = None, section: str = None, question_number: str = None, chunk_type: str = None, top_k: int = 3):
    # Pass doc_id as-is; load_chunks_from_db handles None to mean "all"
    chunks = load_chunks_from_db(doc_id)
    result = answer_bangla_exam_query(query, chunks, section, question_number, chunk_type, top_k)
    return JSONResponse(content=result)
    return JSONResponse(content=result)

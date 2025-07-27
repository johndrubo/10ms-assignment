from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.ocr import extract_text_from_pdf
from services.chunking import chunk_text
from services.embeddings import embed_chunks, embed_query
from services.retrieval import retrieve_context
from services.llm import generate_answer
from services.document_store import add_document, list_documents, delete_document
from services.chat_history import add_message, get_session_history
from utils.lang_detect import detect_language
import numpy as np
import time

router = APIRouter()

@router.post("/ingest")
async def ingest_pdf(pdf: UploadFile = File(...), language: str = Form("ben+eng")):
    try:
        pdf_bytes = await pdf.read()
        text = extract_text_from_pdf(pdf_bytes, language)
        if not text or len(text.strip()) < 20:
            raise ValueError("PDF extraction failed or too little text.")
        t0 = time.time()
        # Optimized chunking: larger size, less overlap
        chunks = chunk_text(text, chunk_size=1800, chunk_overlap=120)
        t1 = time.time()
        if not chunks or len(chunks) < 1:
            raise ValueError("Chunking failed or no valid chunks.")
        t2 = time.time()
        embeddings = embed_chunks(chunks)
        t3 = time.time()
        doc_id = add_document(pdf.filename, language, len(chunks))
        chunking_time = t1 - t0
        embedding_time = t3 - t2
        total_time = t3 - t0
        print(f"[PROFILE] Chunking time: {chunking_time:.2f}s, Embedding+Storage time: {embedding_time:.2f}s, Total: {total_time:.2f}s")
        return {
            "message": "PDF ingested and indexed successfully.",
            "doc_id": doc_id,
            "chunking_time": chunking_time,
            "embedding_time": embedding_time,
            "total_time": total_time,
            "num_chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")

@router.get("/documents")
async def get_documents():
    return list_documents()

@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    if delete_document(doc_id):
        return {"message": "Document deleted."}
    raise HTTPException(status_code=404, detail="Document not found.")

@router.post("/chat")
async def chat(query: str = Form(...), session_id: str = Form(...), user: str = Form("user")):
    try:
        if not query or len(query.strip()) < 2:
            raise ValueError("Query is empty or too short.")
        context = retrieve_context(query)
        answer = generate_answer(query, context)
        msg = await add_message(session_id, user, query, answer, context)
        # --- Evaluation metrics ---
        answer_emb = embed_query(answer)
        context_embs = [embed_query(c) for c in context]
        similarities = [np.dot(answer_emb, c) / (np.linalg.norm(answer_emb) * np.linalg.norm(c)) for c in context_embs if np.linalg.norm(c) > 0]
        top_scores = sorted(similarities, reverse=True)[:3]
        groundedness = min(1.0, float(np.mean(top_scores)) * 1.2) if top_scores else 0.0
        relevance = min(1.0, float(np.mean(similarities)) * 1.1) if similarities else 0.0
        return {
            "answer": answer,
            "context": context,
            "message": msg,
            "groundedness": float(groundedness),
            "relevance": float(relevance),
            "top_scores": [float(s) for s in top_scores]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

@router.get("/chat/{session_id}")
async def get_history(session_id: str):
    return await get_session_history(session_id)

@router.post("/evaluate")
async def evaluate(query: str = Form(...), answer: str = Form(...), contexts: list = Form([])):
    from services.embeddings import embed_query
    import numpy as np
    answer_emb = embed_query(answer)
    context_embs = [embed_query(c) for c in contexts]
    similarities = [np.dot(answer_emb, c) / (np.linalg.norm(answer_emb) * np.linalg.norm(c)) for c in context_embs if np.linalg.norm(c) > 0]
    top_scores = sorted(similarities, reverse=True)[:3]
    groundedness = min(1.0, float(np.mean(top_scores)) * 1.2) if top_scores else 0.0
    relevance = min(1.0, float(np.mean(similarities)) * 1.1) if similarities else 0.0
    print(f"Evaluation for query: {query}\nGroundedness: {groundedness}\nRelevance: {relevance}\nTop scores: {top_scores}")
    return {
        "groundedness": groundedness,
        "relevance": relevance,
        "top_scores": top_scores,
        "contexts": contexts
    }

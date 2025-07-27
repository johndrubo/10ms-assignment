from typing import List, Dict, Optional
from pydantic import BaseModel

class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    language: str
    chunk_count: int

class ChatMessage(BaseModel):
    session_id: str
    user: str
    query: str
    answer: str
    context: List[str]
    timestamp: str

class EvaluationMetrics(BaseModel):
    groundedness: float
    relevance: float
    retrieval_count: int

class ChunkMeta(BaseModel):
    doc_id: str
    section: Optional[str] = None  # e.g., 'মূল আলোচ্য বিষয়', 'প্রাক-মূল্যায়ন'
    question_number: Optional[str] = None  # e.g., '১', '2', etc.
    chunk_type: Optional[str] = None  # e.g., 'passage', 'question', 'annotation', 'mcq', 'word_meaning'
    marks: Optional[int] = None
    extra: Optional[Dict] = None  # For any additional metadata

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    embedding: Optional[List[float]] = None
    meta: ChunkMeta

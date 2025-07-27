import re
from utils.bangla_preprocessing import clean_bangla_text
from utils.lang_detect import detect_language
import numpy as np
from utils.normalization import normalize_bengali_text
import os
from app.models import ChunkMeta, DocumentChunk
import uuid
from services.embeddings import embed_query
import logging
from tqdm import tqdm

# Suppress pymongo DEBUG logs, including heartbeat
logging.getLogger("pymongo").setLevel(logging.WARNING)

# Default chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50

BENGALI_SENT_END = '\u0964'  # Bengali danda
EN_SENT_END = r'[.!?]'  # English sentence endings


def split_sentences(text):
    # Split by Bengali and English sentence boundaries
    # Bengali: danda (\u0964), English: . ! ?
    pattern = rf'(?<={BENGALI_SENT_END})|(?<={EN_SENT_END})'
    sentences = re.split(pattern, text)
    # Remove empty and strip
    return [s.strip() for s in sentences if s.strip()]


def split_paragraphs(text):
    # Split by double newlines or large whitespace
    return [p.strip() for p in re.split(r'\n{2,}|\r{2,}', text) if p.strip()]


def recursive_char_chunking(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, min_chunk_size=MIN_CHUNK_SIZE):
    # Fallback: character-based chunking with overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def semantic_split_by_topic(sentences, chunk_size=1000, chunk_overlap=200, min_chunk_size=50, topic_threshold=0.7, min_topic_len=2):
    # Improved: avoid tiny topic chunks, allow topic_threshold tuning
    if not sentences:
        return []
    chunks = []
    current_chunk = [sentences[0]]
    current_emb = embed_query(sentences[0])
    for sent in tqdm(sentences[1:], desc="Chunking", ncols=80):
        sent_emb = embed_query(sent)
        sim = np.dot(current_emb, sent_emb) / (np.linalg.norm(current_emb) * np.linalg.norm(sent_emb))
        # Only split if chunk is long enough and topic drifts
        if (sim < topic_threshold and len(current_chunk) >= min_topic_len) or len(' '.join(current_chunk)) > chunk_size:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
            # Overlap: keep last N words
            overlap = ''
            if chunk_overlap > 0 and current_chunk:
                words = chunk_text.split()
                overlap_words = words[-max(1, chunk_overlap // 6):]
                overlap = ' '.join(overlap_words)
            current_chunk = [overlap, sent] if overlap else [sent]
            current_emb = embed_query(sent)
        else:
            current_chunk.append(sent)
            # Update embedding as mean of chunk
            chunk_embs = [embed_query(s) for s in current_chunk]
            current_emb = np.mean(chunk_embs, axis=0)
    if current_chunk:
        chunk = ' '.join(current_chunk)
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
    # Remove accidental duplicates
    seen = set()
    final_chunks = []
    for c in chunks:
        if c not in seen:
            final_chunks.append(c)
            seen.add(c)
    return final_chunks


def paragraph_chunking(text, chunk_size=1000, chunk_overlap=200, min_chunk_size=50):
    paragraphs = split_paragraphs(text)
    chunks = []
    current_chunk = []
    current_len = 0
    for para in tqdm(paragraphs, desc="Chunking", ncols=80):
        para_len = len(para)
        if current_len + para_len > chunk_size and current_chunk:
            chunk = ' '.join(current_chunk)
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            # Overlap: keep last N words
            overlap = ''
            if chunk_overlap > 0:
                words = chunk.split()
                overlap_words = words[-max(1, chunk_overlap // 6):]
                overlap = ' '.join(overlap_words)
            current_chunk = [overlap, para] if overlap else [para]
            current_len = len(' '.join(current_chunk))
        else:
            current_chunk.append(para)
            current_len += para_len
    if current_chunk:
        chunk = ' '.join(current_chunk)
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
    return chunks


def chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200,
    min_chunk_size=50,
    topic_aware=True,
    add_metadata=True,
    topic_threshold=0.7,
    min_topic_len=2,
    max_chunks=None,
    min_sentence_len=5,
    max_sentence_len=300,
    deduplicate=True,
    paragraph_mode=False,  # New: allow paragraph-aware chunking
    context_enrichment=True  # New: enrich context for vague queries
):
    lang = detect_language(text)
    if lang == 'bn':
        text = clean_bangla_text(text)
    # Filter sentences by length for better chunking
    sentences = [s for s in split_sentences(text) if min_sentence_len <= len(s) <= max_sentence_len]
    chunks = []
    if paragraph_mode:
        chunks = paragraph_chunking(text, chunk_size, chunk_overlap, min_chunk_size)
    elif topic_aware:
        chunks = semantic_split_by_topic(sentences, chunk_size, chunk_overlap, min_chunk_size, topic_threshold, min_topic_len)
    else:
        # Try sentence-based chunking first
        current_chunk = []
        current_len = 0
        for sent in tqdm(sentences, desc="Chunking by sentence"):
            sent_len = len(sent)
            if current_len + sent_len > chunk_size and current_chunk:
                chunk = ' '.join(current_chunk)
                if len(chunk) >= min_chunk_size:
                    chunks.append(chunk)
                # Overlap: keep last N chars, avoid broken words
                overlap = ''
                if chunk_overlap > 0:
                    words = chunk.split()
                    overlap_words = words[-max(1, chunk_overlap // 6):]
                    overlap = ' '.join(overlap_words)
                current_chunk = [overlap, sent] if overlap else [sent]
                current_len = len(' '.join(current_chunk))
            else:
                current_chunk.append(sent)
                current_len += sent_len
        # Add last chunk
        if current_chunk:
            chunk = ' '.join(current_chunk)
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
    # Fallbacks
    if len(chunks) < 2:
        para_chunks = paragraph_chunking(text, chunk_size, chunk_overlap, min_chunk_size)
        for c in para_chunks:
            if c not in chunks:
                chunks.append(c)
    if len(chunks) < 2:
        char_chunks = recursive_char_chunking(text, chunk_size, chunk_overlap, min_chunk_size)
        for c in char_chunks:
            if c not in chunks:
                chunks.append(c)
    # Deduplicate
    if deduplicate:
        seen = set()
        final_chunks = []
        for c in chunks:
            if c not in seen:
                final_chunks.append(c)
                seen.add(c)
    else:
        final_chunks = chunks
    # Context enrichment for vague queries (optional, stub)
    if context_enrichment and len(final_chunks) < 2:
        # Could add logic to merge small chunks or add extra context
        if len(final_chunks) == 1 and len(final_chunks[0]) < chunk_size // 2:
            final_chunks[0] += '\n' + text[:chunk_size]
    if max_chunks is not None and len(final_chunks) > max_chunks:
        final_chunks = final_chunks[:max_chunks]
    if add_metadata:
        return [
            {
                'text': c,
                'chunk_index': i,
                'length': len(c),
                'language': lang
            } for i, c in enumerate(final_chunks)
        ]
    else:
        return [{'text': c} for c in final_chunks]


def bangla_exam_chunker(text: str, doc_id: str) -> list:
    # Patterns for section headers, question numbers, and annotation blocks
    section_pattern = r'(মূল আলোচ্য বিষয়|প্রাক-মূল্যায়ন|শব্দার্থ ও টীকা|মূল শব্দ|মূল গন্স|মূল প্রশ্ন|উত্তর|উপসংহার)'
    question_pattern = r'([০-৯]+|\d+)[\.|\)]'
    annotation_pattern = r'(\bশব্দার্থ ও টীকা\b|\bমূল শব্দ\b)'
    # Split by section headers
    sections = re.split(section_pattern, text)
    chunks = []
    current_section = None
    for i in range(1, len(sections), 2):
        current_section = sections[i].strip()
        section_text = sections[i+1]
        # Split section into questions/annotations
        q_splits = re.split(question_pattern, section_text)
        for j in range(1, len(q_splits), 2):
            q_num = q_splits[j].strip()
            q_text = q_splits[j+1].strip()
            # Check for annotation
            if re.search(annotation_pattern, q_text):
                chunk_type = 'annotation'
            elif len(q_text) < 100:
                chunk_type = 'question'
            else:
                chunk_type = 'passage'
            chunk_id = str(uuid.uuid4())
            meta = ChunkMeta(
                doc_id=doc_id,
                section=current_section,
                question_number=q_num,
                chunk_type=chunk_type,
                marks=None,
                extra=None
            )
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=q_text,
                embedding=None,
                meta=meta
            )
            chunks.append(chunk)
    return chunks

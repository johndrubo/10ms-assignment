# Bengali/English RAG Chatbot

A modern, scalable Retrieval-Augmented Generation (RAG) chatbot for Bengali/English PDF question answering.

## Features

- Modular FastAPI backend
- Advanced OCR pipeline (Tesseract, pdf2image)
- Language-aware sentence/paragraph chunking with overlap
- Multilingual embeddings (OpenAI or HuggingFace, configurable)
- Extensible LLM integration (Gemini, OpenAI, etc.)
- RESTful API for chat, ingestion, evaluation, and admin
- Robust error handling and validation
- Automated tests
- Environment-based configuration

## Getting Started

1. Clone the repo
2. Create a `.env` file (see `.env.example`)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the FastAPI server: `uvicorn app.main:app --reload`

## Directory Structure

- `app/` - FastAPI endpoints
- `services/` - Core logic (OCR, chunking, embeddings, retrieval, LLM)
- `utils/` - Utility functions (normalization, error handling)
- `tests/` - Automated tests

## Used Tools, Libraries, and Packages

- **FastAPI**: Web API framework
- **Uvicorn**: ASGI server
- **pdf2image**: PDF to image conversion
- **pytesseract**: OCR (Bangla & English)
- **Pillow**: Image processing
- **sentence-transformers**: Multilingual embeddings
- **openai**: (Optional) LLM integration
- **pymongo**: MongoDB vector storage
- **pytest**: Testing
- **langid**: Language detection
- **numpy**: Vector math

## Sample Queries and Outputs

**Bangla Example:**

- Q: `অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?`
- A: `শুম্ভুনাথ`

**English Example:**

- Q: `Who is called a true gentleman in Anupam's words?`
- A: `Shumbhunath`

**API Example:**

```bash
curl -X POST "http://localhost:8000/chat" -F "query=অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?" -F "session_id=test1"
```

## API Documentation

### POST /ingest

Upload and index a PDF document for retrieval and QA.

- **Endpoint:** `/ingest`
- **Method:** POST
- **Request:**
  - `pdf` (file, required): The PDF file to ingest.
  - `language` (form, optional, default: "ben+eng"): Language(s) for OCR.
- **Response:**
  - `message`: Status message
  - `doc_id`: Unique document ID
  - `chunking_time`: Time taken for chunking (seconds)
  - `embedding_time`: Time taken for embedding (seconds)
  - `total_time`: Total processing time (seconds)
  - `num_chunks`: Number of chunks created

**Example:**

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "pdf=@HSC26-Bangla1st-Paper.pdf" \
  -F "language=ben+eng"
```

### POST /chat

Ask a question and get an answer from the indexed documents.

- **Endpoint:** `/chat`
- **Method:** POST
- **Request:**
  - `query` (form, required): The user question (Bangla or English)
  - `session_id` (form, required): Session identifier
  - `user` (form, optional, default: "user"): User name
- **Response:**
  - `answer`: The generated answer
  - `context`: List of retrieved context chunks
  - `message`: Chat message metadata
  - `groundedness`: Groundedness score (float, 0-1)
  - `relevance`: Relevance score (float, 0-1)
  - `top_scores`: Top similarity scores

**Example:**

```bash
curl -X POST "http://localhost:8000/chat" \
  -F "query=অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?" \
  -F "session_id=test1"
```

## Evaluation Matrix

- **Groundedness**: Cosine similarity between the answer and retrieved context (max score per QA)
- **Relevance**: Cosine similarity between the query and retrieved context (max score per QA)
- Both metrics are printed and asserted in tests (threshold: 0.3)

## Answers to Required Questions

**1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**

- We used a two-step process for robust text extraction from educational PDFs, which often contain complex layouts, mixed scripts, and noisy scans. First, we used `pdf2image` to convert each PDF page into a high-resolution image. This approach is more reliable than direct PDF text extraction for Bangla/English exam papers, which frequently use non-standard fonts, tables, and scanned images. Second, we applied `pytesseract` (Tesseract OCR) with both Bangla and English language models enabled (`lang="ben+eng"`). This allows the system to accurately recognize mixed-script content, which is common in Bangladeshi educational materials.

  Formatting challenges included:

  - Noisy backgrounds, faded or skewed scans, and handwritten annotations.
  - Mixed Bangla and English text, sometimes within the same sentence or paragraph.
  - OCR artifacts such as stray pipes, repeated punctuation, and broken diacritics.
  - Inconsistent use of section headers, question numbers, and annotation blocks.

  To address these, we implemented advanced image preprocessing (grayscale conversion, contrast enhancement, binarization, sharpening) and robust postprocessing/normalization (removal of noise, correction of diacritics, and normalization of Bangla script). This pipeline significantly improved OCR accuracy and downstream chunking quality.

**2. What chunking strategy did you choose? Why do you think it works well for semantic retrieval?**

- Our chunking logic is multi-stage and highly adaptive, designed specifically for the challenges of Bangla/English educational texts and exam papers. The pipeline works as follows:

  1. **Preprocessing**: All text is normalized and cleaned using advanced Bangla preprocessing (removal of noise, normalization of diacritics, dialect, and spelling correction) to ensure chunk boundaries are meaningful and not corrupted by OCR artifacts.

  2. **Sentence Splitting**: The text is first split into sentences using both Bangla (danda, Unicode \u0964) and English (., !, ?) sentence boundaries. This ensures that each chunk contains semantically coherent units, regardless of language.

  3. **Topic-Aware Chunking**: The default strategy is topic-aware, sentence-based chunking with overlap. Sentences are grouped into chunks of a target size (e.g., 1000 characters) with a configurable overlap (e.g., 200 characters) to preserve context across chunk boundaries. The system uses semantic similarity (via embeddings) between sentences to detect topic shifts: if the similarity between the current sentence and the chunk drops below a threshold, a new chunk is started. This helps keep each chunk topically focused, which is crucial for accurate retrieval.

  4. **Paragraph and Character Fallbacks**: If the text is not well-structured or the topic-aware method produces too few chunks, the system falls back to paragraph-based chunking (splitting on double newlines) or, as a last resort, character-based chunking with overlap. This ensures that even poorly formatted or OCR-damaged documents are chunked robustly.

  5. **Metadata and Deduplication**: Each chunk is tagged with metadata (index, length, language) and deduplicated to avoid storing or retrieving redundant information. The chunking logic is also exam-aware, with special handling for section headers, question numbers, and annotation blocks in Bangla exam papers (see `bangla_exam_chunker`).

  6. **Context Enrichment**: For very short or vague queries, the system can enrich context by merging small chunks or including additional surrounding text, improving the chances of retrieving relevant information.

- This multi-stage, adaptive chunking strategy works well for semantic retrieval because it:

  - Preserves semantic and topical boundaries, so each chunk is meaningful on its own.
  - Maintains context across chunk boundaries via overlap, which helps answer questions that span multiple sentences or paragraphs.
  - Handles noisy, mixed-language, and poorly formatted documents robustly.
  - Allows for metadata-aware retrieval and evaluation, supporting advanced use cases like exam QA and section-based search.

- The chunking logic is fully configurable and can be tuned for different document types, languages, and retrieval needs. See `services/chunking.py` for implementation details and options.

**3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**

- We selected the `intfloat/multilingual-e5-base` model from the `sentence-transformers` library. This model is specifically designed for multilingual semantic search and supports both Bangla and English natively. It is trained on a large, diverse corpus and is capable of capturing deep semantic relationships between sentences, even across languages. This is essential for our use case, where questions and answers may be in different scripts or contain code-switching. The model produces dense vector representations (embeddings) that encode the meaning of the text, allowing for robust similarity comparison and retrieval. We chose this model for its balance of accuracy, speed, and multilingual support, and because it is open-source and easy to integrate with our Python backend.

**4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**

- Both user queries and document chunks are embedded using the same multilingual model. We use cosine similarity to compare the query embedding with each stored chunk embedding. Cosine similarity is a standard metric for measuring the angle between two vectors in high-dimensional space, making it ideal for semantic search tasks. It is robust to differences in vector magnitude and focuses on the direction (semantic meaning) of the vectors. For storage, we use MongoDB to persist both the chunk text and its embedding, enabling scalable, persistent, and efficient retrieval. This setup allows us to handle large document collections and perform fast similarity search, while also supporting metadata-based filtering and advanced retrieval strategies.

**5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**

- To ensure meaningful comparison, all text (both queries and chunks) undergoes the same preprocessing and normalization pipeline before embedding. This includes language detection, script normalization, noise removal, and spelling correction, which helps align the representation of similar concepts even if they appear in different forms. The chunking strategy is designed to maximize semantic coherence within each chunk, and the retrieval logic can use metadata (e.g., section, question number) to further refine results. If a query is vague or lacks context, the system may retrieve less relevant chunks, as the embedding may not capture enough specific information. In such cases, the context enrichment logic can help by expanding the search to include more surrounding text, but ultimately, vague queries remain a challenge for any retrieval system. Future improvements could include query expansion, user clarification prompts, or more advanced LLM-based re-ranking.

**6. Do the results seem relevant? If not, what might improve them?**

- In our evaluation, the results are generally highly relevant for well-formed, context-rich queries, especially for factual or direct questions from the exam papers. The combination of advanced chunking, robust preprocessing, and a strong multilingual embedding model ensures that the system can retrieve and answer most questions accurately. However, there is always room for improvement. Potential enhancements include:

  - Further improving OCR quality, especially for low-quality scans or handwritten content.
  - Developing even more sophisticated chunking strategies, such as dynamic chunk sizing or LLM-assisted segmentation.
  - Expanding the training data or using larger, more powerful embedding models.
  - Integrating LLM-based answer generation and re-ranking for more nuanced or open-ended questions.
  - Adding user feedback loops to continuously refine retrieval and answer quality.

- Overall, the current system provides a strong foundation for RAG-based educational QA in Bangla and English, and is designed to be easily extensible for future research and production use.

---

This project is designed to fix common pitfalls in RAG chatbot implementations. See code comments for details.

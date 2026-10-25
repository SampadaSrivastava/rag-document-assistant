# Resume and interview notes

Everything below describes functionality that exists in this repository. No metrics are claimed because none were measured.

## Resume bullets
- Built a RAG-based PDF question-answering app in Python using Streamlit, LangChain, ChromaDB, sentence-transformers embeddings and the Gemini API, with page-level source citations and expandable excerpts for every answer.
- Designed an ingestion and retrieval pipeline (PyMuPDF extraction, page-aware recursive chunking, cosine top-k retrieval with a similarity threshold and per-document balancing) supporting multi-PDF questions and an explicit "not found" path that skips the LLM when nothing relevant is retrieved.
- Implemented upload validation, user-friendly handling of API-key, rate-limit, network and corrupted/scanned-PDF failures, secret redaction in logs, and a pytest suite (unit tests plus headless Streamlit UI tests) for the RAG components.

## 2-line description
A Streamlit app that lets users upload PDFs and ask questions answered by Gemini from retrieved ChromaDB chunks, with page-level citations.
It handles multiple documents, summaries and common failure cases, and is configured for deployment on Streamlit Community Cloud.

## Keywords
Retrieval-Augmented Generation, LangChain, ChromaDB, vector embeddings, semantic search, sentence-transformers, Gemini API, prompt engineering, PyMuPDF, Streamlit

## 60-second explanation
"DocuMind AI is a RAG app for asking questions about PDFs. When you upload files, I extract text per page with PyMuPDF, split it into overlapping chunks of about a thousand characters, turn each chunk into an embedding with a small MiniLM model, and store them in ChromaDB with the filename and page number. When you ask a question, I embed it the same way, fetch the closest chunks, and send only those to Gemini with a prompt that says to answer from the context and to say so if the answer isn't there. The UI then shows the answer plus the document and page it came from, with the excerpts. If nothing relevant is retrieved, the app says it couldn't find the answer without even calling the model."

## 2-minute explanation
"The problem is that an LLM doesn't know your documents and will happily guess. RAG fixes that by retrieving evidence first.

The pipeline has two halves. Ingestion: PyMuPDF extracts text page by page, I clean it, and a recursive splitter makes roughly 1000-character chunks with 150 characters of overlap. I split per page so every chunk belongs to one page, which makes citations exact. Each chunk is embedded with all-MiniLM-L6-v2, a small CPU-friendly model, and stored in ChromaDB with metadata: filename, page, document ID and chunk ID. The document ID is a hash of the file contents, so duplicate uploads are detected.

Querying: the question is embedded with the same model, and Chroma returns the nearest chunks by cosine similarity, filtered to the documents the user selected. I use top-5 with a similarity threshold, and with several documents I search each separately so a long document can't crowd out a short one, which matters for something like comparing a resume to a job description. The chunks go into a prompt kept in its own module: it separates instructions, context and question, tells the model to use only the context, never invent details, and reply with a fixed sentence if the answer isn't present. I detect that sentence and hide sources in that case.

Around that, I added validation for corrupted, password-protected, scanned and duplicate PDFs, friendly errors for invalid keys, rate limits and network failures, and tests with a fake embedder and fake LLM so they run without any API key. ChromaDB is isolated in one file, so moving to Pinecone or Qdrant is a contained change. The main limitations are no OCR, no follow-up question handling, and session-only storage on Streamlit Cloud; I haven't measured retrieval accuracy, and a next step would be a small evaluation set to tune chunk size and the threshold."

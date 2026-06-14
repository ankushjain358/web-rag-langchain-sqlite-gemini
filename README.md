# web-rag-langchain-sqlite-gemini

A minimal proof-of-concept for **Retrieval-Augmented Generation (RAG)** on web pages — using LangChain, SQLite + `sqlite-vec`, and Google Gemini.

---

## How it works

```
Web page / file
      │
      ▼
  Extract text
      │
      ▼
  Split into chunks  ──►  Generate embeddings (Gemini)
                                    │
                                    ▼
                             Store in SQLite
                          (sqlite-vec + chunks)

  User query
      │
      ▼
  Embed query (Gemini)
      │
      ▼
  KNN vector search  ──►  Retrieve top-3 chunks
                                    │
                                    ▼
                         Prompt LLM with context
                                    │
                                    ▼
                               Answer ✓
```

---

## Prerequisites

- Python 3.9+
- A free Google Gemini API key from [Google AI Studio](https://ai.google.dev/studio)

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/ankushjain358/web-rag-langchain-sqlite-gemini.git
cd web-rag-langchain-sqlite-gemini

# 2. Create a virtual environment and install dependencies
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your API key
export GOOGLE_API_KEY="your_google_api_key"       # Windows: set GOOGLE_API_KEY=...

# 4. Navigate to the source code directory
cd src 

# 5. Initialize the database
database.py

# 6. Ingest a web page or local file
python ingest.py        # prompts you for a URL or file path

# 7. Ask a question
python search.py        # prompts you for a natural-language query
```

> **Optional:** Run `python sqlite_demo.py` from `src/` to verify your SQLite + sqlite-vec setup before ingesting real data.

---

## Project structure

```
src/
├── database.py     # Initialize SQLite DB with sqlite-vec
├── ingest.py       # Fetch → chunk → embed → store
├── search.py       # Query → embed → KNN search → LLM answer
└── sqlite_demo.py  # Sanity-check script for the DB setup
```

---

## Notes

- Chunks default to **1000 characters** with a **200-character overlap**.
- KNN search retrieves the **top 3** nearest chunks by default.
- If you switch Gemini embedding models, update `VECTOR_DIMENSIONS` in `src/database.py` to match.
- The LLM is instructed to answer only from the retrieved context and say so when the answer isn't there.
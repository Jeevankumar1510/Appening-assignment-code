# Agentic AI Reader

A small FastAPI application for asking questions about `Ebook-Agentic-AI.pdf`. It extracts and indexes the PDF on the first question, retrieves matching passages, and returns an answer with its source passages and a relevance score. The browser interface is served by the same app; no separate frontend build is needed.

## Requirements

- Python 3.11 is verified with this project.
- An authorized copy of `Ebook-Agentic-AI.pdf` must be placed in the project root before asking questions. The supplied eBook and assignment PDFs are intentionally excluded from Git because they may be copyrighted or private source materials.
- Dependencies are listed in `requirements.txt`.

## Run on Windows

Open PowerShell in the project directory. Use the virtual environment's Python directly so install and run commands use the same interpreter:

```powershell
Set-Location C:\Users\Jeeva\Downloads\Chatbot
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\python.exe main.py
```

If `.venv` has not been created yet, create it with Python 3.11 first:

```powershell
py -3.11 -m venv .venv
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\python.exe main.py
```

Use `python -m pip`, not `pip`, if Windows Application Control blocks `pip.exe`.

When Uvicorn starts, open:

- Chat interface: `http://localhost:8000/`
- Interactive API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Stop the server with `Ctrl+C`. If port 8000 is already occupied, stop the old server in its terminal or start this instance on another port:

```powershell
\.venv\Scripts\python.exe -m uvicorn agentic_rag.api:app --host 127.0.0.1 --port 8001
```

Then use `http://localhost:8001/`.

## Ask a question

The homepage provides suggested questions and a text box. Submit with the send button. The page displays the answer, relevance score, and expandable retrieved passages.

The API can also be called directly:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/ask `
  -ContentType 'application/json' `
  -Body '{"query":"What is Agentic AI?"}'
```

Every successful `/ask` response has this shape:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "According to the ebook, Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "retrieved_context_chunks": ["Passages retrieved from the PDF appear here."],
  "confidence_score": 0.9321
}
```

The score is a retrieval relevance heuristic, not a statistically calibrated probability. Unsupported questions return a refusal, no matching context, and a score of `0.0`.

## Request flow

1. The browser sends `{ "query": "..." }` to `POST /ask`.
2. FastAPI validates the request with the `QueryRequest` model in `agentic_rag/api.py`.
3. `answer_query` in `agentic_rag/pipeline.py` ingests the PDF once per running process if it has not already been loaded.
4. `ingest_pdf` extracts text page by page, divides it into overlapping chunks, and records each chunk's source and page number.
5. A LangGraph workflow runs a retrieval node followed by a generation node. The retriever ranks matching chunks; the response builder selects a grounded passage or refuses when evidence is missing.
6. FastAPI serializes the query, answer, retrieved passages, and score as JSON. The homepage renders these fields for the reader.

## Code map

- `main.py` — starts Uvicorn and serves the FastAPI application on port 8000.
- `agentic_rag/api.py` — defines the homepage (`GET /`), health check (`GET /health`), and question endpoint (`POST /ask`).
- `agentic_rag/static/index.html` — self-contained responsive UI, styles, and browser code that calls `/ask`.
- `agentic_rag/config.py` — defines the PDF path and defaults for chunk size, overlap, top-k, and Pinecone index name.
- `agentic_rag/pipeline.py` — PDF extraction, chunking, vector storage, retrieval, answer selection, scoring, and LangGraph workflow.
- `tests/test_rag.py` — checks chunking, retrieval, answer grounding, out-of-scope behavior, and homepage rendering.
- `requirements.txt` — Python packages required by the application and tests.
- `.env.example` — example configuration variables.

## Pipeline details

### PDF ingestion

The source file is resolved from the project root. On the first `/ask` request, each PDF page is extracted with `pypdf`; text is chunked using a default size of 900 characters with 150 characters of overlap. Each chunk stores its text, source path, page number, and ID. The loaded flag prevents re-ingestion on every request; restarting the app clears the local in-memory index and the next question loads the PDF again.

### Retrieval and storage

Without `PINECONE_API_KEY`, chunks are kept in an in-memory store for the lifetime of the server process. If a Pinecone key is set, the pipeline attempts to upsert and query the named Pinecone index; the index must exist and use 128 dimensions. Errors in those Pinecone operations switch the process to the in-memory store.

The current vector values are deterministic token-hash features, not embeddings from a trained language model. Retrieval also requires query-term overlap and uses a small rule-based boost for definition questions. The score describes how well the retrieved evidence matches the query; it does not measure factual certainty or replace human evaluation.

### Answer generation and scope

The current generator is extractive: it selects a relevant sentence from retrieved eBook passages and formats it as an answer. It does not call an LLM. When no passage matches, it returns a refusal rather than using unrelated fallback text. The capital-of-France sample is explicitly treated as out of scope.

## Configuration

`agentic_rag/config.py` reads these optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` | `900` | Maximum chunk length in characters. |
| `CHUNK_OVERLAP` | `150` | Repeated characters between neighboring chunks. |
| `TOP_K` | `3` | Maximum number of matching passages returned. |
| `PINECONE_INDEX` | `agentic-ai-rag` | Pinecone index name when Pinecone is enabled. |
| `PINECONE_API_KEY` | unset | Enables the Pinecone storage path when provided. |

These values must be set in the process environment or VS Code launch configuration. The current code does not automatically load a `.env` file, even though `python-dotenv` is installed.

`OPENAI_API_KEY` and `OPENAI_MODEL` are shown in `.env.example` but are not used by the current implementation.

## Run tests

```powershell
\.venv\Scripts\python.exe -m pytest -q
```

The suite covers chunk overlap, response structure, positive retrieval for an eBook definition, rejection of unrelated chunks, and the homepage route.

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT_DIR / "Ebook-Agentic-AI.pdf"
INDEX_NAME = os.getenv("PINECONE_INDEX", "agentic-ai-rag")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))

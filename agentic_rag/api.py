from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .pipeline import answer_query

app = FastAPI(title="Agentic AI RAG API")
CHAT_PAGE = Path(__file__).parent / "static" / "index.html"


class QueryRequest(BaseModel):
    query: str


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QueryRequest) -> dict[str, object]:
    return answer_query(request.query)

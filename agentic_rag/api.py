from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .pipeline import answer_query

app = FastAPI(title="Agentic AI RAG API")
CHAT_PAGE = Path(__file__).parent / "static" / "index.html"


class QueryRequest(BaseModel):
    query: str


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(
        content=(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<rect width="64" height="64" rx="12" fill="#244f3d"/>'
            '<text x="32" y="46" fill="#fffefa" font-family="Georgia,serif" '
            'font-size="44" text-anchor="middle">A</text></svg>'
        ),
        media_type="image/svg+xml",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QueryRequest) -> dict[str, object]:
    return answer_query(request.query)

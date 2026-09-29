from agentic_rag.pipeline import InMemoryVectorStore, answer_query, build_response, chunk_text
from fastapi.testclient import TestClient

from agentic_rag.api import app

client = TestClient(app)


def test_chunk_text_has_expected_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 3
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert chunks[0][:3] == "abc"


def test_build_response_refuses_out_of_scope_question():
    payload = build_response(
        query="What is the capital of France?",
        context=["Agentic AI is about autonomous decision making."],
        answer="I don't have that information in the provided ebook context.",
        confidence=0.0,
    )

    assert payload["query"] == "What is the capital of France?"
    assert "not available in the ebook context" in payload["final_answer"].lower()
    assert payload["confidence_score"] == 0.0


def test_answer_query_retrieves_ebook_context_with_relevance_score():
    payload = answer_query("What is Agentic AI?")

    assert payload["retrieved_context_chunks"]
    assert any("at its core" in chunk.lower() for chunk in payload["retrieved_context_chunks"])
    assert "refers to systems capable of autonomous decision-making" in payload["final_answer"].lower()
    assert 0.0 < payload["confidence_score"] <= 1.0


def test_vector_search_excludes_chunks_without_query_term_overlap():
    vector_store = InMemoryVectorStore()
    vector_store.upsert(
        ["Autonomous systems perceive their environment and take actions."],
        [{"chunk_text": "Autonomous systems perceive their environment and take actions."}],
    )

    assert vector_store.similarity_search("What is the capital of France?") == []


def test_homepage_renders_chat_interface():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Agentic AI" in response.text
    assert "fetch('/ask'" in response.text

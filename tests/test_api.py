from fastapi.testclient import TestClient


def test_ask_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_MODEL", "hashing")
    monkeypatch.setenv("QDRANT_URL", "")
    from rag import api
    api.assistant.cache_clear()
    client = TestClient(api.app)

    health = client.get("/health").json()
    assert health["status"] == "ok" and health["documents"] > 300

    body = client.post("/ask", json={
        "question": "Quels ordres de modification ont été approuvés pour la pompe "
                    "hydraulique entre août et octobre 2025 ?"}).json()
    assert body["parsed"]["filters"]["part_number"] == "P-1023"
    for src in body["sources"]:
        assert src["status"] == "approved" and "2025-08-01" <= src["date"] <= "2025-10-31"
    assert body["answer"]

    assert client.post("/ask", json={"question": "x"}).status_code == 422

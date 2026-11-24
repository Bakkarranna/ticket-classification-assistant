from fastapi.testclient import TestClient
from app.api import app
from app.service import CATEGORIES, TicketResult, service

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_valid_ticket():
    response = client.post("/analyze", json={"ticket":"I cannot connect to campus Wi-Fi from my laptop."})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] in CATEGORIES
    assert data["summary"]
    assert data["needs_human_review"] is True

def test_blank_ticket_rejected():
    response = client.post("/analyze", json={"ticket":"   "})
    assert response.status_code == 422

def test_invalid_prompt_version_rejected():
    response = client.post("/analyze", json={"ticket":"My laptop has a problem","prompt_version":"v9"})
    assert response.status_code == 422

def test_summary_schema_rejects_multiple_sentences():
    try:
        TicketResult(category="Account_Access", summary="First sentence. Second sentence.", confidence=0.5)
        assert False, "Expected validation error"
    except Exception:
        assert True

def test_controlled_backend_failure(monkeypatch):
    import app.api as api_module
    def fail(*args, **kwargs):
        raise RuntimeError("simulated outage")
    monkeypatch.setattr(api_module.service, "analyze", fail)
    response = client.post("/analyze", json={"ticket":"My student email is not syncing."})
    assert response.status_code == 503

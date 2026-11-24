from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from app.service import TicketRequest, TicketResult, service

app = FastAPI(title="University IT Ticket Classification Assistant", version="1.0.0",
              description="Offline local-mock ticket triage API. Human review is required.")

@app.get("/health")
def health():
    return {"status": "ok", "mode": "local_mock"}

@app.post("/analyze", response_model=TicketResult)
def analyze(payload: TicketRequest):
    try:
        return service.analyze(payload.ticket, payload.prompt_version)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Analysis backend unavailable: {type(exc).__name__}")

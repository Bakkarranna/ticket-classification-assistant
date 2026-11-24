# University IT Ticket Classification Assistant (Week 5 PSED)

## Project overview
A university IT help-desk assistant that accepts a fictional/anonymized ticket, suggests exactly one of ten categories, produces a grounded one-sentence summary, validates the response with Pydantic, and presents it in a Streamlit frontend connected to a FastAPI backend.

**Mode: offline deterministic local mock.** No hosted LLM is called and no model is described as trained. The prompt v1/v2 comparison is implemented as reproducible local similarity/rule fixtures so the project runs without an API key. The prompt files are ready-to-adapt instruction templates for a hosted LLM integration. Reported v1/v2 scores are mock/fixture results, not real LLM performance.

## Scenario and categories
Scenario: University IT help desk.

The ten category labels and their one-line definitions/examples are in `categories.json`:
1. Account_Access
2. Network_Connectivity
3. Email_Services
4. Learning_Platform
5. Student_Portal
6. Software_Installation
7. Hardware_Device
8. Cybersecurity_Report
9. Printing_Services
10. Library_Tech_Services

## Dataset
`dataset.csv` contains 120 original fictional tickets: 100 development/reference rows (10 per category) and 20 fixed evaluation rows (2 per category). The split column is set before evaluation. Evaluation labels should remain unchanged after predictions are inspected. There is no class imbalance: each category has 10 development and 2 evaluation tickets.

## Methods
- **Keyword baseline:** counts category-specific keyword hits and applies a small security override. It is fast and explainable, but fails on synonyms, ambiguous words, and tickets with no known keywords.
- **TF-IDF/cosine similarity:** converts development tickets into TF-IDF vectors, averages vectors per category, and assigns the closest category by cosine similarity. It captures shared terms and phrases, but may miss paraphrases and can be confused by overlapping vocabulary.
- **Prompt v1 and v2:** prompt text is included in `prompts/`. The offline demo approximates a simple v1 similarity decision and a v2 decision with a security override and ambiguity fallback. This is a fixture-based prompt-version demonstration, not an LLM evaluation.

## Output validation and API
`app/service.py` defines:
- `TicketRequest` and `TicketResult` Pydantic models
- a fixed Literal category set of exactly ten values
- nonblank ticket and length checks
- one-sentence summary validation
- reusable local service, keyword baseline and TF-IDF similarity inference

`app/api.py` exposes:
- `GET /health`
- `POST /analyze`

Blank/invalid requests return FastAPI validation errors (HTTP 422). Unavailable analysis backend returns HTTP 503. The UI displays connection and timeout errors.

## Setup
Use Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run locally
Open two terminals from the project root.

Terminal 1, start the API:
```bash
uvicorn app.api:app --reload
```
API docs: http://127.0.0.1:8000/docs

Terminal 2, start Streamlit:
```bash
streamlit run app/streamlit_app.py
```
Frontend: http://localhost:8501

To use a different API address, set `API_URL` before starting Streamlit:
```bash
# Windows PowerShell
$env:API_URL="http://127.0.0.1:8000"
# macOS/Linux
export API_URL="http://127.0.0.1:8000"
```

## Run evaluation
```bash
python evaluation/run_evaluation.py
```
This writes `evaluation_predictions.csv`, accuracy metrics, confusion matrices, classification reports and v2 error rows into `evaluation/`. The evaluation set is the same 20 held-out tickets for all methods.

## Run tests
```bash
pytest -q
```

## Required demo evidence
1. **Valid input:** start both services and submit a ticket such as "I cannot connect to campus Wi-Fi from my laptop." Capture the category and summary.
2. **Blank input:** submit an empty text box; Streamlit should show a validation message.
3. **Backend failure:** stop FastAPI while keeping Streamlit open, then analyze a nonblank ticket. Streamlit should show "Backend unavailable".
4. **Invalid API input:** send an empty ticket to `/analyze` using the API docs or curl; expect HTTP 422.
5. **Controlled backend failure:** the pytest test monkeypatches the service to raise an exception and expects HTTP 503.

Screenshots are not fabricated. Capture them on the machine where the app is run and place them in `screenshots/`.

## Safety and limitations
- All ticket data is fictional; no real student names, IDs, passwords or personal data are included.
- Suggestions are for human review only. The app does not reset passwords, change accounts, issue refunds, or perform device actions.
- The local mock is not a production classifier or a trained model. Similarity scores are not calibrated probabilities.
- A production version should connect a hosted LLM or a supervised classifier, log model/version details, evaluate on a larger independently labeled dataset, and add authentication/rate limiting.
- Prompt injection is addressed in the prompt templates by treating ticket text as untrusted data. The local deterministic mock does not execute ticket instructions.

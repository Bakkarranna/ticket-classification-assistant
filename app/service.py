from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field, field_validator, ConfigDict

CATEGORIES = [
    "Account_Access", "Network_Connectivity", "Email_Services", "Learning_Platform",
    "Student_Portal", "Software_Installation", "Hardware_Device", "Cybersecurity_Report",
    "Printing_Services", "Library_Tech_Services"
]
Category = Literal[
    "Account_Access", "Network_Connectivity", "Email_Services", "Learning_Platform",
    "Student_Portal", "Software_Installation", "Hardware_Device", "Cybersecurity_Report",
    "Printing_Services", "Library_Tech_Services"
]

class TicketRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    ticket: str = Field(min_length=5, max_length=2000)
    prompt_version: Literal["v1", "v2"] = "v2"
    mode: Literal["local_mock"] = "local_mock"

    @field_validator("ticket")
    @classmethod
    def ticket_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ticket text cannot be blank.")
        return value

class TicketResult(BaseModel):
    category: Category
    summary: str = Field(min_length=8, max_length=240)
    mode: str = "local_mock"
    needs_human_review: bool = True
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("summary")
    @classmethod
    def summary_one_sentence(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Summary cannot be blank.")
        if len(re.findall(r"[.!?]", value)) > 1:
            raise ValueError("Summary must be one sentence.")
        return value

KEYWORDS = {
    "Account_Access": ["login", "log in", "password", "locked", "sign-in", "sign in", "authentication", "verification code", "username", "account access"],
    "Network_Connectivity": ["wi-fi", "wifi", "wireless", "internet", "vpn", "network", "ethernet", "connection", "connectivity"],
    "Email_Services": ["email", "mailbox", "outlook", "webmail", "inbox", "mail app", "messages", "spam folder"],
    "Learning_Platform": ["lms", "course page", "assignment", "quiz", "lecture recording", "course site", "learning platform", "coursework", "online test"],
    "Student_Portal": ["student portal", "timetable", "enrollment", "enrolment", "fee challan", "semester result", "registration", "student profile", "attendance record"],
    "Software_Installation": ["install", "installer", "license", "licence", "matlab", "python", "visual studio", "software", "application crashes"],
    "Hardware_Device": ["keyboard", "projector", "monitor", "mouse", "power on", "charging port", "usb port", "desktop computer", "scanner", "screen remains black"],
    "Cybersecurity_Report": ["phishing", "suspicious", "malware", "compromised", "ransomware", "unauthorized", "unfamiliar login", "fake", "credentials", "security warning"],
    "Printing_Services": ["printer", "printing", "print queue", "print credits", "print station", "paper jam", "print job", "pages"],
    "Library_Tech_Services": ["library", "e-journal", "e-journal", "e-book", "database", "catalogue", "catalog", "e-resources", "research paper", "subscribed academic"]
}

def keyword_baseline(ticket: str) -> str:
    text = ticket.lower()
    scores = {cat: sum(1 for kw in words if kw in text) for cat, words in KEYWORDS.items()}
    # Security signals take precedence over generic account/email wording.
    if any(x in text for x in ["phishing", "suspicious", "malware", "ransomware", "compromised", "fake university", "did not initiate"]):
        return "Cybersecurity_Report"
    return max(CATEGORIES, key=lambda c: (scores[c], -CATEGORIES.index(c)))

class LocalTicketService:
    """Offline deterministic demonstration; this is not a trained LLM."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._fit_texts = []
        self._fit_labels = []
        self._fit_matrix = None
        self._fit()

    def _fit(self):
        examples = {
            "Account_Access": ["cannot log in password rejected account locked reset password authentication verification code sign in username"],
            "Network_Connectivity": ["campus wifi wireless internet vpn ethernet network connection disconnects no internet"],
            "Email_Services": ["university email mailbox outlook webmail inbox sending receiving messages synchronization"],
            "Learning_Platform": ["LMS course page assignment upload quiz lecture recording coursework online test learning portal"],
            "Student_Portal": ["student portal timetable enrollment registration fee challan semester result attendance profile"],
            "Software_Installation": ["software install installer license matlab python application update activation"],
            "Hardware_Device": ["hardware computer desktop keyboard projector monitor mouse scanner power charging port screen"],
            "Cybersecurity_Report": ["phishing suspicious email malware compromised account ransomware fake login unauthorized security alert"],
            "Printing_Services": ["printer printing print queue print credits release station paper jam pages print job"],
            "Library_Tech_Services": ["library digital library database e-journal e-book catalogue catalog research paper e-resources"]
        }
        self._fit_labels = list(examples)
        self._fit_texts = [examples[c][0] for c in self._fit_labels]
        self._fit_matrix = self.vectorizer.fit_transform(self._fit_texts)

    def analyze(self, ticket: str, prompt_version: str = "v2") -> TicketResult:
        clean = " ".join(ticket.split())
        q = self.vectorizer.transform([clean])
        sims = cosine_similarity(q, self._fit_matrix)[0]
        order = sims.argsort()[::-1]
        best_idx = int(order[0])
        best_score = float(sims[best_idx])
        second = float(sims[order[1]]) if len(order) > 1 else 0.0
        tfidf_cat = self._fit_labels[best_idx]
        rule_cat = keyword_baseline(clean)
        # v1 trusts the highest TF-IDF similarity. v2 adds a security override and
        # uses a keyword tie-break when similarity is weak/ambiguous.
        if prompt_version == "v1":
            category = tfidf_cat
        else:
            if any(x in clean.lower() for x in ["phishing", "suspicious", "malware", "ransomware", "compromised", "fake university", "did not initiate"]):
                category = "Cybersecurity_Report"
            elif best_score < 0.08 or (best_score - second) < 0.015:
                category = rule_cat
            else:
                category = tfidf_cat
        summary = self._summary(clean)
        confidence = max(0.0, min(1.0, best_score))
        return TicketResult(category=category, summary=summary, confidence=confidence)

    @staticmethod
    def _summary(ticket: str) -> str:
        # Extractive, grounded fallback: preserves the submitted ticket wording.
        text = ticket.strip().rstrip(".!?")
        if len(text) > 180:
            text = text[:177].rsplit(" ", 1)[0] + "..."
        return text[0].upper() + text[1:] + "."

service = LocalTicketService()

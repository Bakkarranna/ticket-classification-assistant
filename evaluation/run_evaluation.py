import sys, json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.service import CATEGORIES, LocalTicketService, keyword_baseline

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "dataset.csv")
dev = df[df.split == "development"]
ev = df[df.split == "eval"].copy()
service = LocalTicketService()

# Class-centroid-style TF-IDF cosine example fitted only on development tickets.
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
X_dev = vectorizer.fit_transform(dev.ticket)
X_eval = vectorizer.transform(ev.ticket)
centroids = []
for cat in CATEGORIES:
    idx = [i for i, label in enumerate(dev.expected_category) if label == cat]
    centroids.append(X_dev[idx].mean(axis=0))
import numpy as np
centroids = np.vstack([np.asarray(x).ravel() for x in centroids])
tfidf_scores = cosine_similarity(X_eval, centroids)
tfidf_preds = [CATEGORIES[i] for i in tfidf_scores.argmax(axis=1)]

ev["keyword_baseline"] = [keyword_baseline(t) for t in ev.ticket]
ev["tfidf_cosine"] = tfidf_preds
ev["prompt_v1_mock"] = [service.analyze(t, "v1").category for t in ev.ticket]
ev["prompt_v2_mock"] = [service.analyze(t, "v2").category for t in ev.ticket]
ev["summary_v2_mock"] = [service.analyze(t, "v2").summary for t in ev.ticket]
ev.to_csv(ROOT / "evaluation" / "evaluation_predictions.csv", index=False)

metrics = {}
for col in ["keyword_baseline", "tfidf_cosine", "prompt_v1_mock", "prompt_v2_mock"]:
    metrics[col] = float(accuracy_score(ev.expected_category, ev[col]))
(ROOT / "evaluation" / "metrics.json").write_text(json.dumps({
    "evaluation_size": int(len(ev)),
    "development_size": int(len(dev)),
    "category_count": int(df.expected_category.nunique()),
    "results": metrics,
    "interpretation": "All prompt_v1_mock and prompt_v2_mock results are deterministic local fixture/similarity results, not hosted LLM performance."
}, indent=2), encoding="utf-8")

for col in ["keyword_baseline", "tfidf_cosine", "prompt_v1_mock", "prompt_v2_mock"]:
    cm = confusion_matrix(ev.expected_category, ev[col], labels=CATEGORIES)
    pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES).to_csv(ROOT / "evaluation" / f"confusion_{col}.csv")
    if col in ["prompt_v1_mock", "prompt_v2_mock"]:
        report = classification_report(ev.expected_category, ev[col], labels=CATEGORIES, zero_division=0)
        (ROOT / "evaluation" / f"classification_report_{col}.txt").write_text(report, encoding="utf-8")

# Human-readable summary and errors.
best = ev[ev.expected_category != ev.prompt_v2_mock][["id","ticket","expected_category","prompt_v2_mock"]]
best.to_csv(ROOT / "evaluation" / "v2_errors.csv", index=False)

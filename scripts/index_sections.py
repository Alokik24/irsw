import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = "data/index_sections/"
SUMMARIES_DIR = "data/summaries/"

os.makedirs(INDEX_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(INDEX_DIR, "vec.joblib")
MATRIX_PATH = os.path.join(INDEX_DIR, "matrix.joblib")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

def build_index():
    docs = []      # section text
    meta = []      # {file, section}

    for fname in os.listdir(SUMMARIES_DIR):
        if not fname.endswith(".json"):
            continue
        
        path = os.path.join(SUMMARIES_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        
        file_id = j.get("file", fname)

        # pick rewritten first, else extractive
        def pick(section):
            return (
                j.get(f"{section}_summary_rewritten")
                or " ".join(j.get(f"{section}_summary_extractive", []))
                or ""
            )
        
        env = pick("environment")
        soc = pick("social")
        gov = pick("governance")

        if env.strip():
            docs.append(env)
            meta.append({"file": file_id, "section": "ENV", "text": env})

        if soc.strip():
            docs.append(soc)
            meta.append({"file": file_id, "section": "SOC", "text": soc})

        if gov.strip():
            docs.append(gov)
            meta.append({"file": file_id, "section": "GOV", "text": gov})

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(X, MATRIX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print("Indexed", len(meta), "section-level documents.")

def normalize(s):
    return "".join(c.lower() for c in s if c.isalnum())

def search_section(query, allowed_sections, top_k=3, company_filter=None):
    vectorizer = joblib.load(VECTORIZER_PATH)
    X = joblib.load(MATRIX_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    qv = vectorizer.transform([query])
    scores = cosine_similarity(qv, X)[0]

    scored = []
    fallback_candidates = []

    company_norm = normalize(company_filter) if company_filter else None

    for i, (m, s) in enumerate(zip(meta, scores)):
        if m["section"] not in allowed_sections:
            continue

        file_norm = normalize(m["file"])

        if company_norm:
            if company_norm in file_norm:
                fallback_candidates.append(i)
            else:
                continue

        scored.append((i, s))

    # Standard TF-IDF scoring
    if scored:
        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "file": meta[idx]["file"],
                "section": meta[idx]["section"],
                "score": float(sc),
                "text": meta[idx]["text"][:500]
            }
            for idx, sc in scored
        ]

    # Fallback: company matched but TF-IDF found no close text
    if company_norm and fallback_candidates:
        idx = fallback_candidates[0]
        return [
            {
                "file": meta[idx]["file"],
                "section": meta[idx]["section"],
                "score": 0.0,
                "text": meta[idx]["text"][:500]
            }
        ]

    return []

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.build:
        build_index()

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib
import re

# Paths
SUMMARIES_DIR = "data/summaries/"
INDEX_DIR = "data/index/"
os.makedirs(INDEX_DIR, exist_ok=True)

# Files to write
VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.npz")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
SVD_PATH = os.path.join(INDEX_DIR, "svd_transformer.joblib")  # optional

# utility: load summaries
def load_summaries(folder: str) -> List[Dict]:
    items = []
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        p = os.path.join(folder, fname)
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        # prioritize rewritten, then environmental/social/governance extractive groups
        def pick_section(jobj, key_base):
            # look for "<key>_summary_rewritten", then "<key>_summary_extractive" (join)
            rep = jobj.get(f"{key_base}_summary_rewritten")
            if rep:
                return rep
            ext = jobj.get(f"{key_base}_summary_extractive") or jobj.get(f"{key_base}")
            if isinstance(ext, list):
                return " ".join(ext)
            return ext or ""
        env = pick_section(j, "environment")
        soc = pick_section(j, "social")
        gov = pick_section(j, "governance")
        full_text = " ".join([seg for seg in [env, soc, gov] if seg and len(seg.strip())>0])
        if not full_text.strip():
            # fallback: try any top-level text fields
            for v in j.values():
                if isinstance(v, str) and len(v) > 200:
                    full_text = v
                    break
        items.append({
            "file": j.get("file") or fname,
            "env": env,
            "soc": soc,
            "gov": gov,
            "text": full_text
        })
    return items

# helper: simple key-phrases (top tfidf terms per doc)
_token_pattern = re.compile(r"(?u)\b\w\w+\b")
def extract_keyphrases(vectorizer: TfidfVectorizer, doc_vec, topn=5) -> List[str]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    if doc_vec.ndim == 1:
        indices = np.argsort(doc_vec)[::-1][:topn]
    else:
        indices = np.argsort(doc_vec.toarray()[0])[::-1][:topn]
    return feature_names[indices].tolist()

# main: build index
def build_index(n_components_svd: int = 0):
    items = load_summaries(SUMMARIES_DIR)
    docs = [it["text"] for it in items]
    ids = [it["file"] for it in items]

    # Vectorizer: use unigrams + bigrams to improve phrase matching
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        max_df=0.8,
        min_df=1,
        token_pattern=r"(?u)\b\w\w+\b"
    )
    X = vectorizer.fit_transform(docs)
    # Optional dimensionality reduction (fast retrieval) — use 0 to disable
    svd = None
    if n_components_svd and n_components_svd < min(X.shape):
        svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
        X = svd.fit_transform(X)
        joblib.dump(svd, SVD_PATH)

    # save artifacts
    joblib.dump(vectorizer, VECTORIZER_PATH)
    # save matrix with joblib (sparse ok)
    joblib.dump(X, MATRIX_PATH)
    # metadata
    metadata = {"ids": ids, "items": items}
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Indexed {len(ids)} documents. Vectorizer -> {VECTORIZER_PATH}, matrix -> {MATRIX_PATH}")

# search function
def search(query: str, top_k: int = 5) -> List[Dict]:
    vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
    X = joblib.load(MATRIX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    ids = metadata["ids"]
    items = metadata["items"]

    qv = vectorizer.transform([query])
    # if we applied SVD, X may be dense with lower dim; check and project query
    if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] != qv.shape[1]:
        # load svd if present
        if os.path.exists(SVD_PATH):
            svd = joblib.load(SVD_PATH)
            qv = svd.transform(qv)
    # compute cos sim
    # ensure both arrays are dense 2D
    if hasattr(X, "toarray"):
        Xmat = X.toarray()
    else:
        Xmat = X
    qvec = qv if not hasattr(qv, "toarray") else qv.toarray()
    sims = cosine_similarity(qvec, Xmat)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        score = float(sims[i])
        it = items[i]
        # compute top keyphrases for the doc using original vectorizer
        doc_vec = vectorizer.transform([it["text"]])
        kps = extract_keyphrases(vectorizer, doc_vec, topn=6)
        results.append({
            "file": ids[i],
            "score": score,
            "key_phrases": kps,
            "env": it.get("env"),
            "soc": it.get("soc"),
            "gov": it.get("gov"),
            "snippet": (it.get("env") or it.get("text") or "")[:400]
        })
    return results

# small CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build TF-IDF IR index or run simple queries.")
    parser.add_argument("--build", action="store_true", help="Build index from data/summaries")
    parser.add_argument("--query", type=str, default=None, help="Run a query")
    parser.add_argument("--topk", type=int, default=5, help="Top-K results")
    parser.add_argument("--svd", type=int, default=0, help="Optional SVD dimension (0 to disable)")
    args = parser.parse_args()

    if args.build:
        build_index(n_components_svd=args.svd)
    elif args.query:
        res = search(args.query, top_k=args.topk)
        for r in res:
            print("===\nFile:", r["file"], "\nScore:", r["score"])
            print("Key phrases:", r["key_phrases"])
            print("Snippet:", r["snippet"][:300])
    else:
        print("Run with --build to build index or --query 'your query' to search.")

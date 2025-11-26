from intent_classifier import predict as pred_intent
from index_sections import search_section
import re
import os

def load_companies():
    path = "data/companies.txt"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [c.strip() for c in f.readlines() if c.strip()]

COMPANIES = load_companies()

def detect_company(query):
    q = query.lower()

    # Load main names
    companies = load_companies()

    # Load alias file
    alias_path = "data/company_aliases.json"
    aliases = {}
    if os.path.exists(alias_path):
        with open(alias_path, "r", encoding="utf-8") as f:
            aliases = json.load(f)

    # 1. Check aliases first (smart detection)
    for alias, full_name in aliases.items():
        if alias.lower() in q:
            return full_name

    # 2. Direct match with main company names
    for c in companies:
        if c.lower() in q:
            return c

    return None


INTENT_TO_KEYWORDS = {
    "ENV_TARGETS": ["emissions", "targets", "net zero"],
    "ENV_POLICIES": ["climate", "policy", "renewable"],
    "SOC_POLICIES": ["employee", "diversity", "training"],
    "SOC_IMPACT": ["community", "impact", "CSR"],
    "GOV_STRUCTURE": ["board", "committee", "oversight"],
    "GOV_COMPLIANCE": ["audit", "compliance", "ethics"]
}

INTENT_TO_SECTION = {
    "ENV_TARGETS": ["ENV"],
    "ENV_POLICIES": ["ENV"],
    "SOC_POLICIES": ["SOC"],
    "SOC_IMPACT": ["SOC"],
    "GOV_STRUCTURE": ["GOV"],
    "GOV_COMPLIANCE": ["GOV"]
}

def ask(query):
    intent = pred_intent(query)
    section = INTENT_TO_SECTION[intent]
    boosted = query + " " + " ".join(INTENT_TO_KEYWORDS[intent])

    # Detect company
    company = detect_company(query)

    if company:
        print(f"[INFO] Company detected: {company}")
        results = search_section(boosted, section, top_k=3, company_filter=company)
    else:
        print("[INFO] No company specified -> Using global search")
        results = search_section(boosted, section, top_k=3)

    return {
        "query": query,
        "intent": intent,
        "section_lookup": section,
        "company": company or "GLOBAL",
        "results": results
    }


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str)
    args = parser.parse_args()

    output = ask(args.q)
    print(json.dumps(output, indent=4))

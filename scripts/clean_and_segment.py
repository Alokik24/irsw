import os, json, re

RAW_TEXT_PATH = "data/text/"
CLEAN_PATH = "data/clean/"
ESG_PATH = "data/esg_segments/"

os.makedirs(CLEAN_PATH, exist_ok=True)
os.makedirs(ESG_PATH, exist_ok=True)

# STEP 1: CLEAN TEXT
def clean_text(t):
    t = re.sub(r'\n\d+\s*\n', '\n', t)      # remove page numbers
    t = re.sub(r'-\s+', '', t)              # fix hyphen word breaks
    t = re.sub(r'[ \t]+', ' ', t)           # collapse spaces, BUT DO NOT remove newlines
    return t.strip()


# STEP 2 — SAFE PARAGRAPH SPLIT
# (PDFMiner produces blank lines between real paras)
def quick_paragraphs(text):
    # split whenever there are 1+ newlines
    paras = re.split(r'\n+', text)
    return [p.strip() for p in paras if len(p.strip()) > 50]

# STEP 3: ESG SEGMENTATION
def segment_esg(text):
    sections = {"E": [], "S": [], "G": []}

    paras = quick_paragraphs(text)

    # keyword lists (broad + stable across ESG reports)
    E_KEYS = ["climate", "carbon", "emission", "energy", "environment", "sustainab",
              "waste", "net zero", "greenhouse", "renewable"]
    S_KEYS = ["employee", "community", "diversity", "social", "inclusion",
              "health", "safety", "wellbeing", "education", "training", "people"]
    G_KEYS = ["governance", "board", "audit", "ethic", "compliance",
              "oversight", "transparency", "risk management"]

    # skip junk boilerplate paragraphs
    SKIP = [
        "about peak re",     # page title
        "about this report",
        "table of contents"
    ]


    for para in paras:
        p = para.lower()

        # skip generic intro garbage
        if any(skip in p for skip in SKIP):
            continue

        # keyword scores
        e = sum(k in p for k in E_KEYS)
        s = sum(k in p for k in S_KEYS)
        g = sum(k in p for k in G_KEYS)

        # if no ESG signals → skip
        if max(e, s, g) == 0:
            continue

        # assign by strongest ESG signal
        if e >= s and e >= g:
            sections["E"].append(para)
        elif s >= e and s >= g:
            sections["S"].append(para)
        else:
            sections["G"].append(para)

    return sections


# STEP 4: PROCESS EACH TEXT FILE
for file in os.listdir(RAW_TEXT_PATH):
    if not file.endswith(".txt"):
        continue

    with open(os.path.join(RAW_TEXT_PATH, file), "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)

    # save cleaned version
    clean_out = os.path.join(CLEAN_PATH, file)
    with open(clean_out, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # segment into ESG
    seg = segment_esg(cleaned)

    # save structured JSON
    json_out = os.path.join(ESG_PATH, file.replace(".txt", ".json"))
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "file": file,
            "environmental": seg["E"],
            "social": seg["S"],
            "governance": seg["G"]
        }, f, indent=4)

    print("Processed:", file)
import os
import json
import spacy

from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
from sumy.parsers.plaintext import PlaintextParser

# PATHS
ESG_PATH = "data/esg_segments/"
OUTPUT_PATH = "data/summaries/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# LOAD SPACY
nlp = spacy.load("en_core_web_sm")

def to_sentences(paragraphs):
    sents = []
    for p in paragraphs:
        doc = nlp(p)
        for s in doc.sents:
            if len(s.text.strip()) > 40:
                sents.append(s.text.strip())
    return sents


# TEXTRANK SUMMARY
LANG = "english"

def textrank_summary(sent_list, n=6):
    if len(sent_list) == 0:
        return []

    text = " ".join(sent_list)
    parser = PlaintextParser.from_string(text, Tokenizer(LANG))

    summarizer = TextRankSummarizer()
    summarizer.stop_words = get_stop_words(LANG)

    summary = summarizer(parser.document, n)
    return [str(s) for s in summary]


# PROCESS ALL JSON FILES
for file in os.listdir(ESG_PATH):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(ESG_PATH, file)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    env = data["environmental"]
    soc = data["social"]
    gov = data["governance"]

    env_sents = to_sentences(env)
    soc_sents = to_sentences(soc)
    gov_sents = to_sentences(gov)

    print(file)
    print("ENV:", len(env_sents))
    print("SOC:", len(soc_sents))
    print("GOV:", len(gov_sents))
    print("-" * 40)


    env_summary = textrank_summary(env_sents, n=7)
    soc_summary = textrank_summary(soc_sents, n=7)
    gov_summary = textrank_summary(gov_sents, n=7)

    out_json = {
        "file": file,
        "environment_summary_extractive": env_summary,
        "social_summary_extractive": soc_summary,
        "governance_summary_extractive": gov_summary
    }

    out_path = os.path.join(OUTPUT_PATH, file.replace(".json", "_summary.json"))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=4)

    print("Summarized:", file)

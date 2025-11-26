# ESG Retrieval & Summarization Engine

A modular NLP pipeline for extracting, summarizing, classifying, and retrieving ESG information.

## 1. Overview
```
Question → Company Detection → Intent Classification → Section Mapping → TF‑IDF Retrieval → Snippet Output
```

## 2. Features
- ESG segmentation
- Extractive + rewritten summaries
- Intent classifier
- Company alias detection
- Section‑level TF‑IDF retrieval
- Fallback retrieval logic
- ROUGE & F1 evaluation
- Automated test suite

## 3. Structure
```
data/
scripts/
tests/
companies.txt
README.md
```

## 4. Setup
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5. Build Steps
```
python scripts/extract_text.py
python scripts/clean_and_segment.py
python scripts/summarize.py
python scripts/index_ir.py
python scripts/index_sections.py --build 
```

## 6. Intent Classifier
```
python scripts/intent_classifier.py --train
python scripts/intent_classifier.py --eval
```

## 7. Querying
```
python scripts/ask_sectioned.py --q "What are Morgan’s climate commitments?"
```

## 8. Evaluation
```
python scripts/eval_rouge.py --orig <seg> --sum <summary>
```

### Example
```
python scripts/eval_rouge.py --orig data/esg_segments/PeakRe_ESG-Disclosure-Report-2023.json --sum data/summaries/PeakRe_ESG-Disclosure-Report-2023_summary.json
```

```
python tests/run_tests.py
```
#### not all test cases will pass, run separately


## 9. Company Aliases
Edit companies.txt:
```
Peak Re
Honeywell | hon | honeywell international
Infosys
Kraft Heinz | Heinz | Kraft
Morgan Stanley | Morgan
```

## 10. Future Work
- SBERT semantic retrieval
- RAG architecture
- Web UI

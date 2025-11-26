import json
import argparse
from rouge_score import rouge_scorer

def rouge1(summary, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return scorer.score(reference, summary)["rouge1"].fmeasure


def evaluate_file(path_original, path_summary):
    with open(path_original, "r", encoding="utf-8") as f:
        orig = json.load(f)

    with open(path_summary, "r", encoding="utf-8") as f:
        summ = json.load(f)

    results = {}

    for sec in ["environment", "social", "governance"]:
        if sec in ["environment"]:
            ref = " ".join(orig.get("environmental", []))
        else:
            ref = " ".join(orig.get(sec, []))

        # extractive first, fallback to rewritten
        ext = " ".join(summ.get(f"{sec}_summary_extractive", []))
        if not ext.strip():
            ext = summ.get(f"{sec}_summary_rewritten", "")

        if not ref.strip():
            print(f"[SKIP] {sec}: reference empty")
            continue

        if not ext.strip():
            print(f"[SKIP] {sec}: summary empty")
            continue

        score = rouge1(ext, ref)
        results[sec.upper()] = score

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True, help="Path to ESG segmentation JSON")
    parser.add_argument("--sum", required=True, help="Path to summary JSON")
    args = parser.parse_args()

    out = evaluate_file(args.orig, args.sum)
    print(out)

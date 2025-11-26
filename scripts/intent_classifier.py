import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from joblib import dump, load

TRAIN_PATH = "data/intent/train_intents.jsonl"
TEST_PATH = "data/intent/test_intents.jsonl"
MODEL_PATH = "data/intent/intent_model.joblib"

def load_dataset(path):
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            X.append(row["text"])
            y.append(row["label"])
    return X, y

def train():
    X_train, y_train = load_dataset(TRAIN_PATH)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=200))
    ])
    model.fit(X_train, y_train)
    dump(model, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

def evaluate():
    model = load(MODEL_PATH)
    X_test, y_test = load_dataset(TEST_PATH)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    print(json.dumps(report))  # REQUIRED FOR TEST HARNESS
    print("Macro F1:", f1_score(y_test, preds, average="macro"))

def predict(q):
    model = load(MODEL_PATH)
    return model.predict([q])[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.train:
        train()
    elif args.eval:
        evaluate()  # FIXED
    elif args.query:
        print(predict(args.query))

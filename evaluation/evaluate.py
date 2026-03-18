import json
import time
import requests
import numpy as np
import re
import sys

CHAT_URL = "http://localhost:5000/chat"
QUESTIONS_FILE = "evaluation/questions.json"

GROUND_THRESHOLD = 0.30


# ---------------------------
# Utility Functions
# ---------------------------

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text


def word_overlap(answer, context):
    a = set(normalize(answer).split())
    c = set(normalize(context).split())

    if len(a) == 0:
        return 0

    return len(a & c) / len(a)


def groundedness_score(answer, context):
    overlap = word_overlap(answer, context)
    return overlap >= GROUND_THRESHOLD


def citation_accuracy(answer, docs):
    answer_words = set(normalize(answer).split())

    for doc in docs:
        text = doc["content"]
        doc_words = set(normalize(text).split())
        overlap = len(answer_words & doc_words)

        if overlap > 5:
            return True

    return False


def exact_match(answer, gold):
    return gold.lower() in answer.lower()


def partial_match(answer, gold):
    answer_words = set(normalize(answer).split())
    gold_words = set(normalize(gold).split())
    return len(answer_words & gold_words) > 0


def check_server():
    try:
        r = requests.get("http://localhost:5000/health")
        return r.status_code == 200
    except:
        return False


# ---------------------------
# Main Evaluation
# ---------------------------

def run():

    if not check_server():
        print("\nERROR: Flask server is not running.")
        print("Run: python app/app.py\n")
        sys.exit()

    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)

    latencies = []
    grounded_correct = 0
    citation_correct = 0
    exact_correct = 0
    partial_correct = 0

    total = len(questions)

    print("\nRunning evaluation...\n")

    for q in questions:

        question = q["question"]
        gold = q.get("gold", "")

        print("Q:", question)

        start = time.time()

        r = requests.post(
            CHAT_URL,
            json={"question": question}
        )

        latency = time.time() - start
        latencies.append(latency)

        if r.status_code != 200:
            print("ERROR:", r.text)
            continue

        data = r.json()

        answer = data.get("answer", "")
        sources = data.get("sources", [])

        print("Answer:", answer[:100], "...")

        docs = []
        context_text = ""

        for s in sources:
            try:
                with open(s, encoding="utf-8") as f:
                    text = f.read()
                    docs.append({
                        "source": s,
                        "content": text
                    })
                    context_text += text
            except:
                pass

        # Groundedness
        if groundedness_score(answer, context_text):
            grounded_correct += 1
            print("Grounded: YES")
        else:
            print("Grounded: NO")

        # Citation Accuracy
        if citation_accuracy(answer, docs):
            citation_correct += 1
            print("Citation: YES")
        else:
            print("Citation: NO")

        # Exact / Partial Match
        if gold:
            if exact_match(answer, gold):
                exact_correct += 1
            if partial_match(answer, gold):
                partial_correct += 1

        print("Latency:", round(latency, 2), "sec")
        print("-" * 40)

    # Final Metrics
    grounded_pct = grounded_correct / total * 100
    citation_pct = citation_correct / total * 100
    exact_pct = exact_correct / total * 100
    partial_pct = partial_correct / total * 100

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    print("\n========== FINAL RESULTS ==========\n")

    print("Total Questions:", total)
    print()

    print("Groundedness %:", round(grounded_pct, 2))
    print("Citation Accuracy %:", round(citation_pct, 2))
    print("Exact Match %:", round(exact_pct, 2))
    print("Partial Match %:", round(partial_pct, 2))
    print()

    print("Latency p50:", round(p50, 2), "sec")
    print("Latency p95:", round(p95, 2), "sec")

    print("\n===================================\n")


if __name__ == "__main__":
    run()
from flask import Flask, request, jsonify
import time
import os

from dotenv import load_dotenv
load_dotenv()

from app.ingest import ingest
from app.rag import build_rag

app = Flask(__name__)

retriever = None
llm = None
prompt = None


def load_rag():
    global retriever, llm, prompt

    if retriever is None:

        if not os.path.exists("vectorstore") and os.path.exists("data/policies"):
          ingest()

        retriever, llm, prompt = build_rag()


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/chat", methods=["POST"])
def chat():
    
    load_rag()

    question = request.json["question"]

    start = time.time()

    docs = retriever.invoke(question)

    # Fallback if nothing retrieved
    if not docs:
        return jsonify({
            "question": question,
            "answer": "I can only answer questions about company policies.",
            "sources": [],
            "latency_seconds": 0
        })

    def clean_source(source):
        return os.path.basename(source).replace("\\", "/")

    context = "\n\n".join([
        f"Document: {clean_source(doc.metadata.get('source', 'Unknown'))}\nContent: {doc.page_content}"
        for doc in docs
    ])

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(formatted_prompt)

    try:
        answer = response.content
    except AttributeError:
        answer = str(response)

    answer = answer.strip()

    latency = time.time() - start

    sources = list(set(
        doc.metadata.get("source", "Unknown") for doc in docs
    ))

    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources,
        "latency_seconds": latency
    })


@app.route("/")
def home():
    return "<h2>Policy Assistant Running</h2>"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
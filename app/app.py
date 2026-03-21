from flask import Flask, request, jsonify
import time
import os

# 🔥 Prevent CPU / memory overuse (VERY IMPORTANT for Render)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

retriever = None
llm = None
prompt = None
rag_loaded = False


# ✅ Lazy load RAG (only when needed)
def load_rag():
    global retriever, llm, prompt, rag_loaded

    if not rag_loaded:
        print("Loading RAG...")

        # 🔥 Lazy import to avoid startup crash
        from app.rag import build_rag

        # 🚨 Do NOT run ingest in production
        if not os.path.exists("vectorstore"):
            print("WARNING: vectorstore not found")

        retriever, llm, prompt = build_rag()

        rag_loaded = True
        print("RAG loaded.")


# ✅ Health check endpoint
@app.route("/health")
def health():
    return {"status": "ok"}


# ✅ Chat endpoint (core logic)
@app.route("/chat", methods=["POST"])
def chat():
    load_rag()

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    start = time.time()

    docs = retriever.invoke(question)

    # ✅ Fallback if nothing retrieved
    if not docs:
        return jsonify({
            "question": question,
            "answer": "I can only answer questions about company policies.",
            "sources": [],
            "snippets": [],
            "latency_seconds": 0
        })

    def clean_source(source):
        return os.path.basename(source).replace("\\", "/")

    # ✅ Build context with document labels
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
        clean_source(doc.metadata.get("source", "Unknown"))
        for doc in docs
    ))

    snippets = [
        doc.page_content[:200] + "..."
        for doc in docs
    ]

    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources,
        "snippets": snippets,
        "latency_seconds": round(latency, 2)
    })


# ✅ Root → Web UI (REQUIRED for assignment)
@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Policy Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>

<body class="bg-gray-100">

<div class="max-w-3xl mx-auto p-4">

    <h1 class="text-2xl font-bold mb-4">Policy Assistant</h1>

    <div id="chat" class="bg-white p-4 rounded shadow h-[500px] overflow-y-auto space-y-4"></div>

    <div class="mt-4 flex gap-2">
        <input id="question"
               class="flex-1 p-2 border rounded"
               placeholder="Ask a policy question..." />
        <button onclick="ask()"
                class="bg-blue-500 text-white px-4 py-2 rounded">
            Send
        </button>
    </div>

    <div id="loading" class="hidden mt-2 text-gray-500">Thinking...</div>

</div>

<script>

function addMessage(role, text, sources=[], snippets=[]) {

    const chat = document.getElementById("chat");

    let bubble = document.createElement("div");

    bubble.className = role === "user"
        ? "text-right"
        : "text-left";

    let content = `
        <div class="${role === "user" ? "bg-blue-500 text-white" : "bg-gray-200"} 
                    inline-block p-3 rounded max-w-[80%]">
            ${text}
        </div>
    `;

    if (role === "assistant" && sources.length > 0) {
        content += `<div class="text-xs text-gray-500 mt-1">Sources:<br>`;
        sources.forEach((src, i) => {
            content += `${src}<br><i>${snippets[i] || ""}</i><br><br>`;
        });
        content += `</div>`;
    }

    bubble.innerHTML = content;
    chat.appendChild(bubble);

    chat.scrollTop = chat.scrollHeight;
}

async function ask() {

    const input = document.getElementById("question");
    const loading = document.getElementById("loading");

    const question = input.value.trim();

    if (!question) return;

    addMessage("user", question);
    input.value = "";

    loading.classList.remove("hidden");

    try {

        const res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ question })
        });

        const data = await res.json();

        if (!res.ok) throw new Error(data.error || "Server error");

        addMessage("assistant", data.answer, data.sources, data.snippets);

    } catch (err) {

        addMessage("assistant", "⚠️ Error: Unable to get response.");

    } finally {

        loading.classList.add("hidden");

    }
}

document.getElementById("question")
    .addEventListener("keypress", function(e) {
        if (e.key === "Enter") ask();
    });

</script>

</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
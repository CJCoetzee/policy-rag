from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM

from app.config import *

import requests
import os


class GroqLLM(LLM):

    def _call(self, prompt, stop=None):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "temperature": 0,  # deterministic output
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        if response.status_code != 200:
            raise ValueError(f"Groq API error: {response.text}")

        data = response.json()

        if "choices" not in data:
            raise ValueError(f"Unexpected Groq response: {data}")

        return data["choices"][0]["message"]["content"]

    @property
    def _llm_type(self):
        return "groq"


PROMPT_TEMPLATE = """
You are a company policy assistant.

You MUST follow these rules strictly:

1. Answer ONLY using the provided context.
2. If the answer is not explicitly in the context, respond EXACTLY with:
"I can only answer questions about company policies."
3. Give a SHORT and PRECISE answer (1 sentence preferred).
4. DO NOT add explanations or extra wording.
5. ALWAYS include a citation in this format:
(Document Name, Section)
6. Use consistent formatting for numbers and currency (e.g., $40, 12 days).

Context:
{context}

Question:
{question}

Answer:
"""


def build_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    db = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 2}   # reduce from 3–5 → 2
    )

    llm = GroqLLM()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return retriever, llm, prompt
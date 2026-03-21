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
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-8b-instant",
                "temperature": 0,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        if response.status_code != 200:
            raise ValueError(f"Groq API error: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    @property
    def _llm_type(self):
        return "groq"


PROMPT_TEMPLATE = """
You are a company policy assistant.

Rules:
- Only answer using the provided context
- If answer not found say:
"I can only answer questions about company policies."
- Always cite document names
- Keep answers short (1 sentence preferred)

Context:
{context}

Question:
{question}

Answer:
"""


def build_rag():
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate

    db = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=None
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})

    llm = GroqLLM()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return retriever, llm, prompt
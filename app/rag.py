def build_rag():
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate

    # 🚨 DO NOT reload embeddings model in production
    # Use persisted embeddings only

    db = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=None   # 🔥 KEY FIX (no model load)
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 2}
    )

    llm = GroqLLM()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return retriever, llm, prompt
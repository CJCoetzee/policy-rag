import random
import numpy as np
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import *

random.seed(SEED)
np.random.seed(SEED)


def ingest():

    print("Loading documents...")

    docs = []

    for file in Path("data/policies").glob("*.md"):

        loader = TextLoader(str(file))

        docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DIR
    )

    print("Vectorstore created")


if __name__ == "__main__":
    ingest()
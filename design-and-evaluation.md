
# Design and Evaluation

## System Architecture

The system follows a standard Retrieval-Augmented Generation architecture:

1. Documents are ingested and chunked.
2. Chunks are embedded using Sentence Transformers.
3. Embeddings are stored in a Chroma vector database.
4. When a user asks a question:
   - Relevant chunks are retrieved
   - A prompt is constructed with the retrieved context
   - The LLM generates an answer.

## Technology Choices

Flask  
Used as a lightweight web API framework.

LangChain  
Used to simplify retrieval pipelines and prompt formatting.

ChromaDB  
Used as the vector database due to its simplicity and local persistence.

Sentence Transformers  
Used to generate document embeddings.

## Architecture Diagram

User → Flask API → Retriever → Vector DB → Context → LLM → Response

## Evaluation Approach

Evaluation focused on:

- answer relevance
- retrieval accuracy
- latency

Example evaluation questions were created based on the policy documents.

## Evaluation Results
Total Questions: 20

Groundedness:       100.0%

Citation Accuracy:  90.0%

Exact Match:        75.0%

Partial Match:      95.0%

Latency p50:        2.71 sec

Latency p95:        2.98 sec

## Improvements

Future improvements could include:

- better chunking strategies
- hybrid search (BM25 + embeddings)
- stronger evaluation datasets

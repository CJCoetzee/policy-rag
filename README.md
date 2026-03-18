**Architecture**
User প্রশ্ন → Flask API → Retriever (Chroma)
                        ↓
                 Top-K Documents
                        ↓
                Prompt Construction
                        ↓
                   LLM (Groq)
                        ↓
              Answer + Citation
Components

Flask API – Handles requests (/chat)

Chroma Vector DB – Stores embeddings

HuggingFace Embeddings – Converts text → vectors

Groq LLM (LLaMA 3.1) – Generates responses

Custom Prompting – Enforces strict output format


**Setup Instructions**
1. Clone the repo
  git clone https://github.com/CJCoetzee/policy-rag
  cd policy-rag

2. Create virtual environment
  python -m venv venv
  venv\Scripts\activate

3. Install dependencies
  pip install -r requirements.txt

4. Run the application
  python app/app.py

  API will be available at:
  
  http://localhost:5000

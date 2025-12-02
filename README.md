# Smart Document Query  
### Chat with any document — instantly, accurately, beautifully

[![Streamlit App](https://img.shields.io/badge/%F0%9F%94%B4%20Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-app-name.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/rag-app?style=social)](https://github.com/yourusername/rag-app)

> **Upload a PDF, DOCX, or TXT → Ask anything → Get perfect answers in seconds.**  
> No hallucinations. No fluff. Powered by **Retrieval-Augmented Generation (RAG)**.

Live App → https://smartdocquery.streamlit.app/ 

---

## Animated Architecture — Watch the Magic Happen

![Smart Document Query – Animated Architecture](https://i.imgur.com/3nK8vL9.gif)

*See how your document becomes intelligent — from upload to answer in real time*

---

## Why This App Will Blow Your Mind

| You Upload | You Ask | You Get |
|-----------|--------|--------|
| Research papers | “What’s the main conclusion?” | Instant, accurate answer with context |
| Legal contracts | “When is the deadline?” | Exact clause, no guesswork |
| Books & manuals | “Summarize Chapter 5” | Clear, concise summary |
| Meeting notes | “Who is responsible for X?” | Direct answer from the text |

Zero hallucinations. Full accuracy. Lightning fast.

---

## Key Technologies Behind the Magic

| Component              | Technology Used                                  | Why It Matters |
|------------------------|---------------------------------------------------|--------------|
| **Frontend**           | Streamlit + Custom CSS + Font Awesome             | Gorgeous, responsive, dark/light mode |
| **Document Loading**   | LangChain (PyPDFLoader, Docx2txtLoader, TextLoader) | Supports PDF, DOCX, TXT |
| **Text Splitting**     | RecursiveCharacterTextSplitter                    | Smart chunking with overlap |
| **Embeddings**         | `sentence-transformers/all-MiniLM-L6-v2`          | Turns text into 384-dim vectors |
| **Vector Database**    | FAISS (Facebook AI Similarity Search)             | Blazing-fast similarity search |
| **RAG Pipeline**       | LangChain + Custom Prompt Template                | Retrieves only relevant context |
| **LLM**                | Groq Llama-3.1-8B (cloud) or Ollama (local)       | Ultra-fast or fully private |
| **Deployment**         | Streamlit Cloud (free, auto-deploy)               | Live in seconds after push |

---

## What These Terms Actually Mean (Simple & Clear)

| Term               | What It Is                                                                 | Why You Should Care |
|--------------------|-----------------------------------------------------------------------------|---------------------|
| **Embeddings**     | Mathematical fingerprints of text (384 numbers) that capture meaning       | Allows the app to "understand" similarity |
| **Vector Database**| A super-fast search engine for embeddings (FAISS)                          | Finds the most relevant parts of your document in milliseconds |
| **RAG**            | Retrieval-Augmented Generation = Search + LLM                               | Prevents AI from making things up |
| **LLM**            | Large Language Model (like Llama 3.1)                                       | Generates human-like answers |
| **Groq**           | The fastest AI inference in the world (~200 tokens/sec)                     | Answers appear instantly |
| **Ollama**         | Run LLMs locally (100% private, no API key needed)                         | Great for sensitive documents |

---

## How to Use (30 Seconds)

1. **Upload** your document (PDF, DOCX, or TXT)
2. Click **"Load & Index Document"**
3. Type your question → Press Enter
4. **Get perfect answers instantly**

**That’s it.** No training. No setup. No nonsense.

---

## Run Locally (2 Minutes)

```bash
git clone https://github.com/laoluafolami/rag-app.git
cd rag-app
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
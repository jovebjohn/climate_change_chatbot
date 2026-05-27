# Climate Change RAG Chatbot

A RAG-based chatbot built using LangChain, ChromaDB, Ollama, and Streamlit. The chatbot answers questions from PDF documents using semantic search and local LLMs.

## Features
- PDF Question Answering
- Semantic Search with Embeddings
- Chroma Vector Database
- Streamlit Chatbot UI
- CoG Prompting
- Local LLM using Llama3

## Tech Stack
- Python
- LangChain
- Streamlit
- ChromaDB
- Ollama

## Run Project

### Create Vector Database
```bash
python ingest.py
```

### Run Chatbot
```bash
streamlit run app.py
```

### Run Evaluation
```bash
python evaluate.py
```

## Evaluation
The chatbot was evaluated using:
- Accuracy
- Precision
- Recall

## Author
Jove B John
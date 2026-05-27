import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Page title
st.title("Climate Change Chatbot")

# Load embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Load vector database
db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# Load LLM
llm = OllamaLLM(
    model="llama3"
)

# User query
query = st.text_input("Ask a question")

if query:

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    # CoG Prompt
    prompt = f"""
You are an expert climate research assistant.

Step 1:
Understand the user's question.

Step 2:
Analyze the retrieved context carefully.

Step 3:
Reason step-by-step.

Step 4:
Provide a concise and accurate answer.

Rules:
- Use only the provided context
- Do not hallucinate
- If answer is unavailable, say so

Context:
{context}

Question:
{query}

Final Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    # Display output
    st.write("### Answer")
    st.write(response)
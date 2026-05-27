from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Load database
db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = OllamaLLM(model="llama3")

# Chat loop
while True:
    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    # Combine retrieved text
    context = "\n".join([doc.page_content for doc in docs])

    # Prompt
    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

    # Generate response
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response)
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

# --------------------------------------------------
# Vector DB
# --------------------------------------------------

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# --------------------------------------------------
# Ollama LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3",
    temperature=0.2
)

# --------------------------------------------------
# Query
# --------------------------------------------------

query = "what is amazon"
print(f"\nUser Query: {query}")

# --------------------------------------------------

# Retrieval
# --------------------------------------------------

retriever = db.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke(query)

print("\n--- Retrieved Context ---")
for i, doc in enumerate(docs, 1):
    print(f"\nDoc {i}:")
    print(doc.page_content)

# --------------------------------------------------
# Prompt Grounding
# --------------------------------------------------

combined_input = f"""
Answer the question using ONLY the documents below.

Question:
{query}

Documents:
{"\n".join([f"- {doc.page_content}" for doc in docs])}

If the answer is not present, say:
"I don't have enough information to answer that question."
"""

messages = [
    SystemMessage(
    content=(
        "You are an expert assistant. "
        "You must produce clean, structured, and readable answers. "
        "You must ONLY use the provided documents as your source of truth. "
        "Do not add external knowledge."
    )
),

    HumanMessage(content=combined_input),
]

# --------------------------------------------------
# Generation (Ollama)
# --------------------------------------------------

response = llm.invoke(messages)

print("\n--- Generated Answer ---")
print(response.content)

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PERSIST_DIRECTORY = "db/chroma_db"

print("🔁 Loading embedding model (Hugging Face, local)...")

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "Who succeeded Ze'ev Drori as CEO in October 2008?"

print(f"\n❓ User Query: {query}")

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

print("\n--- 🔎 Retrieved Context ---")
for i, doc in enumerate(relevant_docs, start=1):
    print(f"\n📄 Document {i}:")
    print(doc.page_content)



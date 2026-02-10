import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

# 1. Import Hugging Face Embeddings (Free & Local)
from langchain_huggingface import HuggingFaceEmbeddings

# Optional: Load env if you have one, but not needed for local embeddings
load_dotenv()

# 2. Prevent the "LangSmith" logging crash
os.environ["LANGCHAIN_TRACING_V2"] = "false"

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"📂 Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print(f"⚠️ Created missing directory: {docs_path}")
        print("Please add .txt files to this folder and run again.")
        return []
    
    # Load all .txt files with UTF-8 encoding
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        print(f"⚠️ No .txt files found in {docs_path}.")
        return []
    


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    if not documents:
        return []

    print("\n✂️ Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separator="\n"
    )   
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        print(f"✅ Created {len(chunks)} chunks.")
        
        # Preview first chunk
        print(f"\n--- Chunk 1 Preview ---")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print("-" * 50)
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    if not chunks:
        return None

    print(f"\n💾 Creating vector store in '{persist_directory}'...")
    print("   (Downloading free model 'all-MiniLM-L6-v2'... this runs locally)")
        
    # 3. USE HUGGING FACE EMBEDDINGS (Free, runs on CPU)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create ChromaDB vector store
    print("--- Processing Vectors... ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore

# 4. The "Start Button" (Main Execution Block)
if __name__ == "__main__":
    print("=== 🚀 RAG Ingestion Pipeline (Hugging Face Edition) ===")
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # Step 1: Load
    documents = load_documents(docs_path)
    
    # Step 2: Split
    if documents:
        chunks = split_documents(documents)
        
        # Step 3: Embed & Store
        if chunks:
            create_vector_store(chunks, persistent_directory)
    
    print("\n=== Pipeline Finished ===") 
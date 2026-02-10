# RAG Fundamentals – Chunking Strategies

This repository explores **different text chunking strategies** used in modern **Retrieval-Augmented Generation (RAG)** systems.

The goal of this project is to understand **how chunking impacts retrieval quality**, embeddings, and downstream LLM responses — not just to use defaults blindly.

---

## 📌 Why Chunking Matters in RAG

In a RAG pipeline, **chunking happens before embeddings**.

Bad chunking leads to:
- poor semantic embeddings
- irrelevant retrieval
- hallucinated answers

Good chunking improves:
- retrieval accuracy
- context relevance
- answer quality

This repo demonstrates multiple chunking approaches and their trade-offs.

---

## 🧠 Chunking Strategies Implemented

### 1️⃣ Character-Based Chunking
- Splits text purely by character count
- Simple baseline
- Demonstrates why naive chunking fails for real documents

**Use case:** Learning & comparison only

---

### 2️⃣ Recursive Character Chunking (Recommended Default)
- Structure-aware splitting
- Uses multiple separators (paragraphs, sentences, words)
- Deterministic and fast

**Use case:**  
PDFs, documents, policies, most production RAG systems

---

### 3️⃣ Semantic Chunking (Embedding-Based)
- Splits text based on **meaning similarity**
- Uses embeddings to detect topic shifts
- Produces conceptually coherent chunks

**Trade-off:**  
Higher compute cost, less predictable chunk sizes

---

### 4️⃣ LLM-Assisted (Prompt-Based) Chunking
- Uses a local LLM (Ollama) to decide split points
- Chunk boundaries based on human-like understanding
- Highly flexible and expressive

**Note:**  
This is **LLM-assisted processing**, not agentic AI (single-pass, no feedback loop).

---

## 🏗️ Project Structure

```text
.
├── chunking/
│   ├── character_chunking.py
│   ├── recursive_chunking.py
│   ├── semantic_chunking.py
│   ├── llm_chunking.py
├── requirements.txt
├── README.md

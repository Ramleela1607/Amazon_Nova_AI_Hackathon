# 📄 Smart Document Copilot  
### #AmazonNova Hackathon Submission  
Category: Multimodal Understanding

---

## 🚀 Overview

Smart Document Copilot is a multimodal AI application built using **Amazon Nova foundation models on AWS Bedrock**.

It allows users to:

- 📤 Upload PDF documents
- 🧠 Index documents using Amazon Nova Multimodal Embeddings
- 🔎 Ask grounded questions using Retrieval-Augmented Generation (RAG)
- 📚 View source citations for transparency
- 🧾 Extract structured key fields as JSON

The system ensures answers are grounded strictly in the uploaded document, reducing hallucinations and improving reliability.

---

## 🧠 Built With Amazon Nova

This project uses:

- **Amazon Nova Multimodal Embeddings**
  - For generating vector embeddings of document chunks
  - Used in semantic search (FAISS)

- **Amazon Nova Lite (Inference Profile: apac.amazon.nova-lite-v1:0)**
  - For reasoning and generating grounded answers
  - Used for structured JSON extraction

All models are invoked through **Amazon Bedrock**.

---

## 🏗 Architecture

User Upload → PDF Text Extraction → Chunking  
→ Nova Multimodal Embeddings → FAISS Vector Index  
→ Semantic Retrieval → Nova Lite Reasoning  
→ Answer + Source Citations

---

## 📦 Features

### 1️⃣ Document Q&A (Grounded RAG)
- Retrieves top relevant chunks
- Generates answer using only retrieved sources
- Displays citation sources used

### 2️⃣ Structured Information Extraction
- Automatically detects document type
- Extracts key fields
- Returns valid JSON output
- Provides confidence score

---

## 🛠 Tech Stack

- Python
- Streamlit (UI)
- FAISS (Vector Search)
- boto3 (AWS SDK)
- Amazon Bedrock
- Amazon Nova foundation models

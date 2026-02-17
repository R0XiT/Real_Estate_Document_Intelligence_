# 🏗 Real Estate Document Intelligence System

## 📌 Overview

This project implements a working prototype of a **Real Estate Document Intelligence System** that allows users to:

- **Upload** real estate PDFs
- **Convert** documents into searchable embeddings
- **Query** documents using natural language
- **Retrieve** relevant text snippets with metadata (PDF name + page number)
- **Measure** system performance and retrieval quality

The system is designed with:

- **Latency awareness**
- **Scalability considerations**
- **Retrieval accuracy optimization**
- **Production-readiness thinking**

---

## 🧠 System Architecture

```
User → FastAPI → Embedder → FAISS → Reranker → Response
               ↑
        Upload → Extract → Chunk → Embed → Index
```

### Core Components

| Component | Technology |
|-----------|------------|
| PDF Extraction | PyMuPDF |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Index | FAISS (IndexFlatIP) |
| Reranking | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Backend | FastAPI |
| Evaluation | Custom evaluation scripts |

---
## ⚙️ Setup

### 1. Clone Repository

```bash
git clone <repo-link>
cd real_estate_doc_ai

```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install python-multipart

```

### 4. Run the API

```bash
uvicorn api.app:app --reload

```

**Open:**

http://127.0.0.1:8000/docs

---

## 🚀 Usage

### Upload PDF

**POST** `/upload`

Upload a PDF via Swagger UI.

**Example response:**

```json
{
  "message": "file uploaded and indexed successfully",
  "chunks_added": 129,
  "total_vectors": 129
}
```

### Query System

**POST** `/query`

**Request:**

```json
{
  "question": "What parking facilities are available?"
}
```

##Performance Testing

**Latency:**
```bash
python evaluation/latency_test.py
```

**Retrieval Quality:**
```bash
python evaluation/evaluate.py
```

**Response includes:**

- PDF name
- Page number
- Relevance score
- Snippet
- Query latency

---


## 🔄 End-to-End Workflow

### 1️⃣ PDF Upload

- User uploads a PDF via **`/upload`**
- Text extracted using **PyMuPDF**
- Clean paragraph-based chunking applied
- Each chunk stored with:
  - `pdf_name`
  - `page_number`
  - `chunk_id`

### 2️⃣ Embedding Generation

- Each chunk converted into a **384-dimensional** embedding
- Vectors **normalized** for cosine similarity
- Stored inside **FAISS index** for fast retrieval

### 3️⃣ Query Flow

When a query is sent to **`/query`**:

**Step 1 — Vector Retrieval**

- Query converted into embedding
- FAISS retrieves **Top-K** similar chunks

**Step 2 — Cross-Encoder Reranking**

- Query + chunk pairs scored semantically
- **Top 3** reranked results returned
- Each result includes:
  - PDF name
  - Page number
  - Snippet text

---

## ⚡ Performance Metrics

Measured using **20 real estate queries** stored in:

- `evaluation/test_questions.json`

### 📊 Latency Results (20 Queries)

| Metric | Value |
|--------|--------|
| **Total Queries** | 20 |
| **Average Latency** | **0.2458 seconds** |
| **P95 Latency** | **0.4866 seconds** |

✅ **Requirement:** < 2 seconds  
✔ **Achieved comfortably**

---

## 🎯 Retrieval Quality Evaluation

A manually curated evaluation set of **20 real estate questions** was used.

**Evaluation script:** `evaluation/evaluate.py`

**Metrics computed:**

- Top-1 Accuracy
- Top-3 Accuracy
- Average Latency
- P95 Latency

### 📈 Results

| Metric | Value |
|--------|--------|
| **Total Questions** | 20 |
| **Top-1 Accuracy** | **65%** |
| **Top-3 Accuracy** | **85%** |

Reranking significantly improves retrieval precision compared to pure vector search.

---

## 📈 System Behavior & Scalability Analysis

### What Happens as PDFs Grow Larger?

- **More pages** → more chunks
- **More chunks** → larger FAISS index
- **Larger index** → slower search

**Current FAISS type:** `IndexFlatIP` (exact similarity search)  
- **Time complexity:** O(N)

**Suitable for:**

- Prototype scale
- ~10K chunks
- Single-node deployment

### What Would Break First in Production?

1. **RAM Usage** — FAISS stores embeddings in memory. Memory grows linearly.
2. **Cross-Encoder Latency** — Reranking model is transformer-based and CPU-intensive.
3. **Synchronous Upload** — Large PDFs may block request thread during ingestion.

### Bottlenecks

| Area | Limitation |
|------|------------|
| FAISS Flat Index | Linear search |
| Reranker | CPU-heavy |
| JSON Metadata | Not scalable |
| No Caching | Repeated queries recomputed |

### How to Scale Further

- Use **FAISS IVF or HNSW** (sublinear search)
- Move metadata to **SQL/NoSQL database**
- Add **Redis caching**
- Use **async ingestion pipeline**
- Deploy **multiple API workers**

---

## 🏆 Design Decisions & Justifications

| Decision | Justification |
|----------|---------------|
| **Why FAISS Flat Index?** | Exact similarity search, deterministic behavior, easier debugging, appropriate for prototype scale. |
| **Why Reranking?** | Improves semantic matching; boosted Top-3 accuracy significantly. |
| **Why FastAPI?** | Lightweight, async-ready, built-in OpenAPI documentation, clean API interface for testing. |

---

## 🎥 Demo Video

[![Watch Demo](demo_thumbnail.png)](https://github.com/user-attachments/assets/49461576-16bf-47f7-ad77-cf635790c1ef)

[![Watch Demo](demo_thumbnail.png)](https://github.com/user-attachments/assets/a0ce9f4b-e1ab-4c7c-8cbc-f379c0de7297)


---

## 📂 Project Structure

```
real_estate_doc_ai/
│
├── api/
│   └── app.py
│
├── ingestion/
│   ├── pdf_extractor.py
│   └── chunker.py
│
├── index/
│   └── build_index.py
│
├── evaluation/
│   ├── latency_test.py
│   ├── evaluate.py
│   └── test_questions.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📄 PDFs Used for Evaluation

The system was tested using the following real estate brochures:

| Document | Source |
|----------|--------|
| **Max Towers Brochure** | https://maxestates.in/downloads |
| **222 Rajpur Brochure** | https://maxestates.in/downloads |

These PDFs were used to:

- Validate retrieval performance
- Measure latency
- Compute Top-1 and Top-3 accuracy

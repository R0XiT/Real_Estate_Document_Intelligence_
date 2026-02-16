from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import faiss
import numpy as np
import json
import time
import os
import uuid
import fitz
from sentence_transformers import SentenceTransformer, CrossEncoder

# ----------------------------
# Configuration
# ----------------------------

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 10

CHUNK_SIZE = 256
OVERLAP = 50
MIN_TOKENS = 20  # unified threshold

DATA_FOLDER = "data"
INDEX_PATH = "index/faiss_index.index"
METADATA_PATH = "embeddings/metadata.json"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs("index", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

app = FastAPI(title="Real Estate Document Intelligence API")

print("Loading models...")

embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

embedding_dim = embedder.get_sentence_embedding_dimension()

# ----------------------------
# Safe Index Initialization
# ----------------------------

if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")
    index = faiss.read_index(INDEX_PATH)
else:
    print("Creating new FAISS index...")
    index = faiss.IndexFlatIP(embedding_dim)

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = []

print("System Ready.")

# ----------------------------
# Chunking
# ----------------------------

def chunk_text(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(tokens), CHUNK_SIZE - OVERLAP):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]

        if len(chunk_tokens) < MIN_TOKENS:
            continue

        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

# ----------------------------
# Root Endpoint
# ----------------------------

@app.get("/")
def root():
    return {"message": "Real Estate Document Intelligence API is running."}

# ----------------------------
# Query Schema
# ----------------------------

class QueryRequest(BaseModel):
    question: str

# ----------------------------
# Query Endpoint
# ----------------------------

@app.post("/query")
def query_documents(request: QueryRequest):
    start_time = time.time()
    question = request.question

    if index.ntotal == 0:
        return {"message": "No documents indexed yet."}

    query_embedding = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, TOP_K)

    candidate_chunks = [metadata[idx]["text"] for idx in indices[0]]
    candidate_pages = [metadata[idx]["page_number"] for idx in indices[0]]
    candidate_pdfs = [metadata[idx]["pdf_name"] for idx in indices[0]]

    pairs = [[question, chunk] for chunk in candidate_chunks]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(rerank_scores, candidate_pages, candidate_pdfs, candidate_chunks),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for score, page, pdf, text in reranked[:3]:
        results.append({
            "pdf_name": pdf,
            "page_number": page,
            "score": float(score),
            "snippet": text[:300]
        })

    latency = time.time() - start_time

    return {
        "question": question,
        "latency_seconds": round(latency, 4),
        "results": results
    }

# ----------------------------
# Upload Endpoint
# ----------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    file_path = os.path.join(DATA_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    doc = fitz.open(file_path)
    new_chunks = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text").strip()

        if not text:
            continue

        chunks = chunk_text(text, embedder.tokenizer)

        for chunk in chunks:
            new_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "pdf_name": file.filename,
                "page_number": page_number + 1,
                "text": chunk
            })

    if not new_chunks:
        return {"message": "No extractable text found."}

    texts = [chunk["text"] for chunk in new_chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index.add(embeddings)
    metadata.extend(new_chunks)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "message": f"{file.filename} uploaded and indexed successfully.",
        "chunks_added": len(new_chunks),
        "total_vectors": index.ntotal
    }

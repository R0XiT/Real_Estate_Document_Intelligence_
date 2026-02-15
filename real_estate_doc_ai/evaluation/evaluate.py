import faiss
import numpy as np
import json
import time
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_PATH = "index/faiss_index.index"
METADATA_PATH = "embeddings/metadata.json"
EVAL_PATH = "evaluation/test_questions.json"
TOP_K = 10


def evaluate():
    print("Loading models and index...")

    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        evaluation_set = json.load(f)

    top1_hits = 0
    top3_hits = 0
    latencies = []

    print("\nRunning retrieval evaluation...\n")

    for item in evaluation_set:
        question = item["question"]
        relevant_pages = item["relevant_pages"]

        start_time = time.time()

        # Vector retrieval
        query_embedding = embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding, TOP_K)

        candidate_chunks = [metadata[idx]["text"] for idx in indices[0]]
        candidate_pages = [metadata[idx]["page_number"] for idx in indices[0]]

        # Reranking
        pairs = [[question, chunk] for chunk in candidate_chunks]
        rerank_scores = reranker.predict(pairs)

        reranked = sorted(
            zip(rerank_scores, candidate_pages),
            key=lambda x: x[0],
            reverse=True
        )

        reranked_pages = [page for _, page in reranked]

        latency = time.time() - start_time
        latencies.append(latency)

        if reranked_pages[0] in relevant_pages:
            top1_hits += 1

        if any(page in relevant_pages for page in reranked_pages[:3]):
            top3_hits += 1

        print(f"Q: {question}")
        print(f"Top-3: {reranked_pages[:3]}")
        print(f"Relevant: {relevant_pages}")
        print("-" * 50)

    total = len(evaluation_set)

    print("\n================ RESULTS ================")
    print(f"Total Questions: {total}")
    print(f"Top-1 Accuracy: {(top1_hits/total):.2%}")
    print(f"Top-3 Accuracy: {(top3_hits/total):.2%}")
    print(f"Average Latency: {np.mean(latencies):.4f}s")
    print(f"P95 Latency: {np.percentile(latencies,95):.4f}s")
    print("=========================================")


if __name__ == "__main__":
    evaluate()

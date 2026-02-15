import faiss
import numpy as np
import json
import time
from sentence_transformers import SentenceTransformer



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10



def initialize_system():
    print("Initializing system...")
    start_init = time.time()

    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index("index/faiss_index.index")

    with open("embeddings/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    init_time = time.time() - start_init
    print(f"System ready. Initialization took {init_time:.2f} seconds.\n")

    return model, index, metadata



def search(query, model, index, metadata):
    start_time = time.time()

    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Perform search
    scores, indices = index.search(query_embedding, TOP_K)

    search_time = time.time() - start_time

    print(f"\nQuery: {query}")
    print(f"Search completed in {search_time:.4f} seconds")
    print("Top Results:\n")

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        result = metadata[idx]

        print(f"Rank {rank} | Score: {score:.4f}")
        print(f"PDF: {result['pdf_name']}")
        print(f"Page: {result['page_number']}")
        print(f"Snippet: {result['text'][:300]}...")
        print("-" * 80)

    return search_time



if __name__ == "__main__":
    model, index, metadata = initialize_system()

    while True:
        user_query = input("\nEnter your query (or type 'exit'): ")

        if user_query.lower() == "exit":
            print("Exiting...")
            break

        search(user_query, model, index, metadata)

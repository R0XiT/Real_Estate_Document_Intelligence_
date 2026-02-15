import faiss
import numpy as np
import time
import os


def build_faiss_index():
    print("Loading embeddings...")
    
    embeddings = np.load("embeddings/embeddings.npy")

    print(f"Embeddings shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    print("Building FAISS IndexFlatIP...")

    start_time = time.time()

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    build_time = time.time() - start_time

    print(f"Index built in {build_time:.4f} seconds")
    print(f"Total vectors indexed: {index.ntotal}")

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, "index/faiss_index.index")

    print("\nFAISS index saved to index/faiss_index.index")


if __name__ == "__main__":
    build_faiss_index()

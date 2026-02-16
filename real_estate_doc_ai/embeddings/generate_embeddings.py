import json
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def generate_embeddings():
    print("Loading embedding model...")
    start_load = time.time()

    model = SentenceTransformer(MODEL_NAME)

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds\n")

    # Load chunks
    with open("ingestion/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Total chunks to embed: {len(texts)}\n")

    print("Generating embeddings...")
    start_embed = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embed_time = time.time() - start_embed

    print(f"\nEmbedding generation took {embed_time:.2f} seconds")
    print(f"Average time per chunk: {embed_time / len(texts):.4f} seconds")

    # Normalize for cosine similarity search
    faiss.normalize_L2(embeddings)

    # Save embeddings
    np.save("embeddings/embeddings.npy", embeddings)

    # Save metadata
    with open("embeddings/metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("\n Embeddings saved.")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total vectors stored: {embeddings.shape[0]}")


if __name__ == "__main__":
    generate_embeddings()

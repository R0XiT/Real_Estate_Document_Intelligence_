import json
import uuid
import re
from tqdm import tqdm


def chunk_text(text):
   
    text = re.sub(r'\s+', ' ', text)

    paragraphs = re.split(r'(?<=[.!?])\s+', text)

    clean_chunks = []

    for para in paragraphs:
        para = para.strip()

       
        if len(para) < 40:
            continue

        if para.isdigit():
            continue

        clean_chunks.append(para)

    return clean_chunks


def create_chunks():
    with open("ingestion/extracted_pages.json", "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []

    print("Creating chunks...")
    for page in tqdm(pages):
        text = page["text"]

        if not text.strip():
            continue

        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "pdf_name": page["pdf_name"],
                "page_number": page["page_number"],
                "text": chunk
            })

    with open("ingestion/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n Created {len(all_chunks)} chunks.")
    print("Saved to ingestion/chunks.json")


if __name__ == "__main__":
    create_chunks()

import fitz  
import os
import json
from tqdm import tqdm


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text()

        pages.append({
            "pdf_name": os.path.basename(pdf_path),
            "page_number": page_number + 1,
            "text": text
        })

    return pages


def process_all_pdfs(data_folder="data"):
    all_pages = []

    for file in tqdm(os.listdir(data_folder)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, file)
            pages = extract_text_from_pdf(pdf_path)
            all_pages.extend(pages)

    with open("ingestion/extracted_pages.json", "w") as f:
        json.dump(all_pages, f, indent=2)

    print(f"Extracted {len(all_pages)} pages.")


if __name__ == "__main__":
    process_all_pdfs()

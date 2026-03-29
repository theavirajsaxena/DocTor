# extractor.py
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Opens a PDF and extracts text from each page.
    Returns a list of dicts: [{ "page": 1, "text": "..." }, ...]
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction
        text = text.strip()

        if text:  # skip blank pages
            pages.append({
                "page": page_num + 1,  # human-readable page number
                "text": text
            })

    doc.close()
    return pages
# modules/pdf_parser.py
import fitz  # PyMuPDF

def parse_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)

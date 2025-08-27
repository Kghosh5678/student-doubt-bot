import fitz  # PyMuPDF

def parse_pdf_text(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Split text into chunks for embedding
    chunks = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

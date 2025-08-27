from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.pdf_parser import parse_pdf_text
from app.embedder import get_embeddings
from app.vector_store import add_documents, search_similar
from app.qa_engine import answer_question
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs allowed")
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    text_chunks = parse_pdf_text(file_location)
    embeddings = get_embeddings(text_chunks)
    add_documents(text_chunks, embeddings, metadata={"source": file.filename})
    return {"message": f"Uploaded and processed {file.filename}"}

@app.post("/ask/")
async def ask_question(question: str):
    results = search_similar(question, top_k=3)
    answer = answer_question(question, results)
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Student Doubt Bot is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

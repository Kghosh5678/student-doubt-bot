import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_data"
))

collection_name = "student_books"

# Create or get collection
try:
    collection = client.get_collection(collection_name)
except Exception:
    collection = client.create_collection(collection_name)

def add_documents(texts, embeddings, metadata=None):
    metadatas = [metadata or {}] * len(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )
    client.persist()

def search_similar(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    # Returns list of matched documents
    return results['documents'][0]

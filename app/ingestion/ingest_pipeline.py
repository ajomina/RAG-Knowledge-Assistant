from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import create_chunks
from app.ingestion.embedder import generate_embeddings
from app.retrieval.vector_store import (
    create_faiss_index,
    save_index,
)


def ingest_pdf(pdf_path: str):
    pages = load_pdf(pdf_path)

    chunks = create_chunks(
        pages,
        chunk_size=500,
        overlap=100,
    )

    embeddings = generate_embeddings(chunks)

    index = create_faiss_index(embeddings)

    save_index(
        index=index,
        metadata=chunks,
        index_path="data/vector_db/faiss.index",
        metadata_path="data/vector_db/metadata.pkl",
    )

    return len(chunks)
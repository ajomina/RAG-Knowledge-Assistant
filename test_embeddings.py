from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import create_chunks
from app.ingestion.embedder import generate_embeddings
from app.retrieval.vector_store import (
    create_faiss_index,
    save_index,
)

# Step 1: Load the PDF
pages = load_pdf("data/uploads/sample.pdf")

# Step 2: Create chunks
# Recommended values for research papers
chunks = create_chunks(
    pages,
    chunk_size=500,
    overlap=100,
)

print(f"Total chunks: {len(chunks)}")

# Step 3: Generate embeddings
embeddings = generate_embeddings(chunks)

print(f"Embedding shape: {embeddings.shape}")

# Step 4: Create FAISS index
index = create_faiss_index(embeddings)

# Step 5: Save FAISS index and metadata
save_index(
    index=index,
    metadata=chunks,
    index_path="data/vector_db/faiss.index",
    metadata_path="data/vector_db/metadata.pkl",
)

print("FAISS index saved successfully.")
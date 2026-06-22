from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import create_chunks

pages = load_pdf("data/uploads/sample.pdf")

chunks = create_chunks(
    pages,
    chunk_size=100,
    overlap=20,
)

print(f"Total Chunks: {len(chunks)}")

for chunk in chunks:
    print("-" * 40)
    print(f"Chunk ID : {chunk['chunk_id']}")
    print(f"Page     : {chunk['page']}")
    print(f"Text     : {chunk['text']}")
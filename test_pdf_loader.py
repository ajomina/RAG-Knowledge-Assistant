from app.ingestion.pdf_loader import load_pdf

pdf_path = "data/uploads/sample.pdf"

pages = load_pdf(pdf_path)

print(f"Total extracted pages: {len(pages)}")

for page in pages:
    print("-" * 40)
    print(f"Page: {page['page']}")
    print(page["text"][:300])  # Preview first 300 characters
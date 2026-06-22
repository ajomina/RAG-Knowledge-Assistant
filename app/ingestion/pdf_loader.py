from pathlib import Path
import fitz  # PyMuPDF


def load_pdf(pdf_path: str):
    """
    Load a PDF and extract text page by page.

    Returns:
        List[dict]:
        [
            {
                "page": 1,
                "text": "...page text..."
            },
            ...
        ]
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_file)

    pages = []

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "page": page_index + 1,
                    "text": text,
                }
            )

    document.close()

    return pages
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_chunks(
    pages,
    chunk_size=1000,
    overlap=200,
):
    """
    Create semantic chunks from extracted PDF pages.

    Args:
        pages (list): List of page dictionaries
        chunk_size (int): Maximum chunk size
        overlap (int): Overlap between chunks

    Returns:
        list: Chunk dictionaries
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = []
    chunk_id = 1

    for page in pages:

        page_number = page["page"]
        page_text = page["text"]

        if not page_text.strip():
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk_text in page_chunks:

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk_text,
                }
            )

            chunk_id += 1

    return chunks
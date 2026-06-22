from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def generate_embeddings(chunks):
    """
    Generate embeddings for chunk texts.

    Args:
        chunks (list)

    Returns:
        numpy.ndarray
    """

    if not chunks:
        return np.array([])

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.astype(
        "float32"
    )
import numpy as np

from app.ingestion.embedder import model
from app.retrieval.vector_store import load_index


def retrieve(
    query: str,
    top_k: int = 5,
    distance_threshold: float = 1.5,
):
    """
    Retrieve the most relevant chunks from FAISS.

    Args:
        query (str): User question
        top_k (int): Number of chunks to retrieve
        distance_threshold (float): Similarity filter

    Returns:
        list
    """

    index, metadata = load_index(
        "data/vector_db/faiss.index",
        "data/vector_db/metadata.pkl",
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    distances, indices = index.search(
        np.asarray(
            query_embedding,
            dtype="float32",
        ),
        top_k,
    )

    results = []

    for score, idx in zip(
        distances[0],
        indices[0],
    ):

        if idx == -1:
            continue

        chunk = metadata[idx].copy()

        chunk["distance"] = float(score)

        # Keep relevant chunks
        if score <= distance_threshold:
            results.append(chunk)

    # If everything was filtered out,
    # return the best result instead of nothing.
    if not results and len(indices[0]) > 0:

        best_idx = indices[0][0]

        if best_idx != -1:

            chunk = metadata[best_idx].copy()

            chunk["distance"] = float(
                distances[0][0]
            )

            results.append(chunk)

    return results
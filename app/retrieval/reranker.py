from sentence_transformers import CrossEncoder

# Load once at startup
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(
    query: str,
    retrieved_chunks: list,
    top_n: int = 5,
):
    """
    Re-rank retrieved chunks using CrossEncoder.

    Args:
        query (str)
        retrieved_chunks (list)
        top_n (int)

    Returns:
        list
    """

    if not retrieved_chunks:
        return []

    pairs = [
        (query, chunk["text"])
        for chunk in retrieved_chunks
    ]

    scores = reranker_model.predict(pairs)

    reranked = []

    for chunk, score in zip(
        retrieved_chunks,
        scores,
    ):
        updated_chunk = chunk.copy()

        updated_chunk[
            "rerank_score"
        ] = float(score)

        reranked.append(
            updated_chunk
        )

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return reranked[:top_n]
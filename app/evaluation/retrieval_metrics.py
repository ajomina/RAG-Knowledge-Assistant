import math

from app.evaluation.text_utils import (
    chunks_to_text,
    clamp_score,
    keyword_tokens,
    meaningful_tokens,
)


def retrieval_quality(retrieved_chunks, expected_keywords):
    """Estimate retrieval relevance on a 0.0-1.0 scale."""
    if not retrieved_chunks:
        return 0.0

    context_tokens = set(meaningful_tokens(chunks_to_text(retrieved_chunks)))
    query_tokens = keyword_tokens(expected_keywords)

    lexical_score = (
        len(query_tokens & context_tokens) / len(query_tokens)
        if query_tokens
        else 0.0
    )

    distances = [
        float(chunk["distance"])
        for chunk in retrieved_chunks[:5]
        if isinstance(chunk, dict)
        and chunk.get("distance") is not None
        and math.isfinite(float(chunk["distance"]))
    ]
    semantic_score = (
        sum(1.0 / (1.0 + max(distance, 0.0)) for distance in distances)
        / len(distances)
        if distances
        else lexical_score
    )

    score = 0.65 * lexical_score + 0.35 * semantic_score
    return round(clamp_score(score), 3)

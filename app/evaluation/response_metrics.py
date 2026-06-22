from app.evaluation.text_utils import clamp_score, keyword_tokens, meaningful_tokens


NOT_FOUND_ANSWER = "i could not find the answer in the provided documents"


def response_quality(answer, expected_keywords, retrieved_chunks=None):
    """Estimate answer relevance and grounding on a 0.0-1.0 scale."""
    normalized_answer = str(answer or "").strip().lower()
    if not normalized_answer or normalized_answer.startswith("llm error"):
        return 0.0
    if NOT_FOUND_ANSWER in normalized_answer:
        return 0.0

    answer_tokens = set(meaningful_tokens(answer))
    query_tokens = keyword_tokens(expected_keywords)
    if not answer_tokens:
        return 0.0

    relevance_score = (
        len(query_tokens & answer_tokens) / len(query_tokens)
        if query_tokens
        else 0.0
    )

    if retrieved_chunks:
        context_tokens = set(
            meaningful_tokens(
                " ".join(str(chunk.get("text", "")) for chunk in retrieved_chunks)
            )
        )
        grounding_score = len(answer_tokens & context_tokens) / len(answer_tokens)
        score = 0.7 * grounding_score + 0.3 * relevance_score
    else:
        score = relevance_score

    return round(clamp_score(score), 3)

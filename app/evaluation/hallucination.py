from app.evaluation.text_utils import chunks_to_text, meaningful_tokens


NOT_FOUND_ANSWER = "i could not find the answer in the provided documents"


def hallucination_rate(answer: str, retrieved_chunks):
    """Estimate the percentage of answer content tokens unsupported by context."""
    normalized_answer = str(answer or "").strip().lower()
    if not normalized_answer or NOT_FOUND_ANSWER in normalized_answer:
        return 0.0

    answer_tokens = set(meaningful_tokens(answer))
    context_tokens = set(meaningful_tokens(chunks_to_text(retrieved_chunks)))

    if not answer_tokens:
        return 0.0
    if not context_tokens:
        return 100.0

    unsupported = answer_tokens - context_tokens
    return round((len(unsupported) / len(answer_tokens)) * 100.0, 2)

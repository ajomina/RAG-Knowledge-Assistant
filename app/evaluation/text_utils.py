import re


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "may",
    "of", "on", "or", "our", "should", "that", "the", "their", "then",
    "there", "these", "this", "those", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your",
}


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def meaningful_tokens(text) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [_stem(token) for token in tokens if len(token) > 2 and token not in STOPWORDS]


def keyword_tokens(expected_keywords) -> set[str]:
    if isinstance(expected_keywords, str):
        expected_keywords = [expected_keywords]
    return set(meaningful_tokens(" ".join(str(item) for item in (expected_keywords or []))))


def chunks_to_text(retrieved_chunks) -> str:
    return " ".join(
        str(chunk.get("text", ""))
        for chunk in (retrieved_chunks or [])
        if isinstance(chunk, dict)
    )


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

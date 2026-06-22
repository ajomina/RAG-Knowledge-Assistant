from app.retrieval.retriever import retrieve
from app.evaluation.retrieval_metrics import retrieval_quality

query = "What is self-attention?"

results = retrieve(query, top_k=5)

score = retrieval_quality(
    results,
    expected_keywords=[
        "self-attention",
        "attention",
        "sequence",
    ],
)

print("Retrieval Quality Score:", score)
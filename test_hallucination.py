from app.generation.rag_pipeline import ask
from app.retrieval.retriever import retrieve
from app.evaluation.hallucination import hallucination_rate

question = "What is self-attention?"

result = ask(question)
retrieved = retrieve(question, top_k=5)

score = hallucination_rate(
    result["answer"],
    retrieved,
)

print("Estimated Hallucination Rate:", score)
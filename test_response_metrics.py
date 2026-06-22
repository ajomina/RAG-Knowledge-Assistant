from app.generation.rag_pipeline import ask
from app.evaluation.response_metrics import response_quality

question = "What is self-attention?"

result = ask(question)

score = response_quality(
    result["answer"],
    expected_keywords=[
        "self-attention",
        "sequence",
        "representation",
    ],
)

print("Response Quality Score:", score)
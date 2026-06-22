from app.generation.rag_pipeline import ask

question = "What is self-attention?"

result = ask(
    question=question,
    top_k=5,
)

print("\nQUESTION\n")
print(question)

print("\nANSWER\n")
print(result["answer"])

print("\nCITATIONS\n")

for citation in result["citations"]:
    print(
        f"Page {citation['page']} | "
        f"Chunk {citation['chunk_id']}"
    )

    print(citation["source_preview"])
    print("-" * 60)
from app.generation.rag_pipeline import ask

question1 = "What is self-attention?"
result1 = ask(question1)

print("\nQUESTION 1")
print(question1)

print("\nANSWER 1")
print(result1["answer"])

question2 = "Why is it useful?"
result2 = ask(question2)

print("\nQUESTION 2")
print(question2)

print("\nANSWER 2")
print(result2["answer"])
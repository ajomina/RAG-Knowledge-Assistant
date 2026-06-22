from app.retrieval.retriever import retrieve

query = "What is Retrieval-Augmented Generation?"

results = retrieve(query, top_k=3)

print(f"Retrieved {len(results)} chunks\n")

for result in results:
    print("-" * 50)
    print(f"Chunk ID : {result['chunk_id']}")
    print(f"Page     : {result['page']}")
    print(f"Distance : {result['distance']:.4f}")
    print("Text:")
    print(result["text"])
    print()
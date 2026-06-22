from app.retrieval.retriever import retrieve
from app.retrieval.reranker import rerank

query = "What is Retrieval-Augmented Generation?"

retrieved = retrieve(query, top_k=3)

reranked = rerank(query, retrieved)

print("Re-ranked Results:\n")

for item in reranked:
    print("-" * 50)
    print(f"Chunk ID     : {item['chunk_id']}")
    print(f"Page         : {item['page']}")
    print(f"Rerank Score : {item['rerank_score']:.4f}")
    print(item["text"])
    print()
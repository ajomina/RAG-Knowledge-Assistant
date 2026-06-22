from app.retrieval.retriever import retrieve
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import load_index
from app.generation.llm import generate_answer
from app.memory.conversation_memory import ConversationMemory
from app.evaluation.retrieval_metrics import retrieval_quality
from app.evaluation.response_metrics import response_quality
from app.evaluation.hallucination import hallucination_rate

memory = ConversationMemory()


def ask(question: str, top_k: int = 5):

    question_lower = question.lower()

    summary_mode = any(
        keyword in question_lower
        for keyword in [
            "summarize",
            "summary",
            "overview",
            "entire paper",
            "paper summary",
            "explain the paper",
            "conclusion",
            "conclusions",
            "key findings",
            "findings",
            "main contributions",
            "takeaways",
            "results",
        ]
    )

    # =====================================
    # RETRIEVAL
    # =====================================

    if summary_mode:

        _, metadata = load_index(
            "data/vector_db/faiss.index",
            "data/vector_db/metadata.pkl",
        )

        retrieved = sorted(
            metadata,
            key=lambda x: (
                x["page"],
                x["chunk_id"],
            ),
        )[:103]

        for chunk in retrieved:

            chunk["distance"] = 0.0
            chunk["rerank_score"] = 1.0

        reranked = retrieved

    else:

        retrieved = retrieve(
            query=question,
            top_k=top_k,
            distance_threshold=1.5,
        )

        if not retrieved:

            return {
                "answer":
                    "I could not find the answer in the provided documents.",
                "citations": [],
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "evaluation": {
                    "retrieval_quality": 0.0,
                    "response_quality": 0.0,
                    "hallucination_rate": 0.0,
                },
            }

        reranked = rerank(
            question,
            retrieved,
            top_n=5,
        )

        if not reranked:

            return {
                "answer":
                    "I could not find the answer in the provided documents.",
                "citations": [],
                "retrieved_chunks": retrieved,
                "reranked_chunks": [],
                "evaluation": {
                    "retrieval_quality": retrieval_quality(retrieved, question),
                    "response_quality": 0.0,
                    "hallucination_rate": 0.0,
                },
            }
    print("\n========== RETRIEVED ==========\n")

    for chunk in retrieved[:5]:

        print(
            f"PAGE={chunk['page']} "
            f"CHUNK={chunk['chunk_id']}"
        )

        print(chunk["text"][:300])
        print("-" * 50)
    # =====================================
    # CONTEXT
    # =====================================

    context = "\n\n".join(
        chunk["text"]
        for chunk in reranked
    )

    if not context.strip():

        return {
            "answer":
                "I could not find the answer in the provided documents.",
            "citations": [],
            "retrieved_chunks": retrieved,
            "reranked_chunks": reranked,
            "evaluation": {
                "retrieval_quality": retrieval_quality(reranked, question),
                "response_quality": 0.0,
                "hallucination_rate": 0.0,
            },
        }

    # =====================================
    # LLM
    # =====================================

    answer = generate_answer(
        query=question,
        context=context,
        conversation_history=memory.get_history(),
        summary_mode=summary_mode,
    )
    # =====================================
    # EVALUATION
    # =====================================

    retrieval_score = retrieval_quality(
        retrieved_chunks=reranked,
        expected_keywords=question,
    )

    response_score = response_quality(
        answer=answer,
        expected_keywords=question,
        retrieved_chunks=reranked,
    )

    hallucination_score = hallucination_rate(
        answer=answer,
        retrieved_chunks=reranked,
    )
    # =====================================
    # MEMORY
    # =====================================

    memory.add_user_message(question)
    memory.add_assistant_message(answer)

    # =====================================
    # CITATIONS
    # =====================================

    citations = []

    for chunk in reranked[:10]:

        citations.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "source_preview":
                    chunk["text"][:200] + "...",
                "distance":
                    chunk.get("distance"),
                "rerank_score":
                    chunk.get("rerank_score"),
            }
        )

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": retrieved,
        "reranked_chunks": reranked,

        "evaluation": {
            "retrieval_quality": retrieval_score,
            "response_quality": response_score,
            "hallucination_rate": hallucination_score,
        }
    }

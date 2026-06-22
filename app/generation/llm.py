from ollama import chat


def generate_answer(
    query: str,
    context: str,
    conversation_history=None,
    summary_mode=False,
):

    if conversation_history is None:
        conversation_history = []

    history_text = ""

    for msg in conversation_history[-6:]:

        history_text += (
            f"{msg['role'].capitalize()}: "
            f"{msg['content']}\n"
        )

    # ==========================================
    # SUMMARY MODE
    # ==========================================

    if summary_mode:

        prompt = f"""
You are a research paper analysis assistant.

IMPORTANT RULES:

1. Use ONLY the provided CONTEXT.
2. Do NOT use external knowledge.
3. Do NOT invent facts.
4. If a section is unavailable in the context, write:
   "Not explicitly mentioned in the document."
5. Produce a structured report.
6. Be concise but complete.

CONTEXT:
{context}

Create a report using EXACTLY these sections:

# Paper Title

# Objective

# Problem Addressed

# Transformer Architecture

# Self-Attention Mechanism

# Multi-Head Attention

# Positional Encoding

# Datasets Used

# Experimental Results

# BLEU Scores

# Main Contributions

# Conclusion

REPORT:
"""

    # ==========================================
    # QUESTION ANSWERING MODE
    # ==========================================

    else:

        prompt = f"""
You are a document question-answering assistant.

IMPORTANT RULES:

1. Answer ONLY from the provided CONTEXT.
2. Use CONVERSATION HISTORY only to understand follow-up questions.
3. Do NOT use outside knowledge.
4. Do NOT make assumptions.
5. Do NOT invent information.
6. If the answer is not present in the CONTEXT, reply EXACTLY:

I could not find the answer in the provided documents.

7. Keep answers factual and concise.
8. Prefer direct answers.
9. Quote numbers exactly when available.

CONVERSATION HISTORY:
{history_text}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:

        response = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return (
            response["message"]["content"]
            .strip()
        )

    except Exception as e:

        return (
            f"LLM Error: {str(e)}"
        )
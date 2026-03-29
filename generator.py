# generator.py
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "z-ai/glm-4.5-air:free"


def _build_prompt(query: str, chunks: list[dict]) -> str:
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"""
[Passage {i+1} | Page {chunk['page']}]
{chunk['text']}
---"""

    prompt = f"""You are DocTor, an expert assistant for understanding technical documents.

You have been given the following numbered passages retrieved from a document:

{context_block}

STRICT RULES:
1. Answer ONLY using information from the passages above.
2. Do NOT use any outside knowledge or make assumptions.
3. While answering, cite passages inline like this: [1], [2], [3]
   Example: "The system uses a transformer architecture [1] which was
   later improved with attention mechanisms [2]."
4. At the end of your answer, add a "Sources:" section listing each
   cited passage like this:
   Sources:
   [1] Passage 1 — Page 3
   [2] Passage 2 — Page 5
5. If the passages do not contain enough information to answer, say:
   "I could not find sufficient information in the document to answer
   this question."
6. Keep your answer clear, accurate and well-structured.

Question: {query}

Answer:"""

    return prompt


def _extract_cited_passage_numbers(answer_text: str) -> list[int]:
    """
    Scans the answer text for inline citations like [1], [2], [3]
    and returns a list of the passage numbers that were actually cited.
    """
    matches = re.findall(r'\[(\d+)\]', answer_text)
    # Deduplicate while preserving order
    seen = set()
    cited = []
    for m in matches:
        num = int(m)
        if num not in seen:
            seen.add(num)
            cited.append(num)
    return cited


def _highlight_passages(
    answer_text : str,
    chunks      : list[dict]
) -> list[dict]:
    """
    Builds a rich citation object for each passage that was
    actually cited in the answer. Includes the full passage text
    so the frontend can highlight it.
    """
    cited_numbers = _extract_cited_passage_numbers(answer_text)
    citations = []

    for num in cited_numbers:
        idx = num - 1  # passages are 1-indexed in the prompt
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            citations.append({
                "citation_number" : num,
                "chunk_id"        : chunk["chunk_id"],
                "page"            : chunk["page"],
                "retrieval_rank"  : chunk["retrieval_rank"],
                "full_text"       : chunk["text"],
                "preview"         : chunk["text"][:200] + "..."
            })

    return citations


def generate_answer(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer"   : "No relevant chunks were retrieved. Please try a different question.",
            "citations": [],
            "sources"  : []
        }

    prompt = _build_prompt(query, chunks)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        answer_text = response.choices[0].message.content

        # Build citation objects from passages actually cited
        citations = _highlight_passages(answer_text, chunks)

        # Build simple sources list (all retrieved, not just cited)
        sources = []
        for chunk in chunks:
            sources.append({
                "chunk_id"    : chunk["chunk_id"],
                "page"        : chunk["page"],
                "rank"        : chunk["retrieval_rank"],
                "text_preview": chunk["text"][:150] + "..."
            })

        return {
            "answer"   : answer_text,
            "citations": citations,
            "sources"  : sources
        }

    except Exception as e:
        return {
            "answer"   : f"Error generating answer: {str(e)}",
            "citations": [],
            "sources"  : []
        }
"""
chat.py — Interactive RAG question-answering interface.

Flow
----
1. The user types a question.
2. `query_chroma` retrieves the top-k most relevant summary chunks.
3. If OPENAI_API_KEY is set, GPT-4o-mini synthesises an answer grounded in those chunks.
4. If the key is missing or the API call fails, the retrieved chunks are printed
   directly as a graceful local fallback — useful for demos without an API key.

The OPENAI_API_KEY is read from the `.env` file at the repo root via `python-dotenv`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .index import query_chroma

load_dotenv()


def _build_context(contexts: list[dict]) -> str:
    """
    Serialise retrieved Chroma hits into a formatted context block for the LLM prompt.

    Each hit is formatted as:
        [<video_id> <start>-<end>s] <summary text>

    Parameters
    ----------
    contexts:
        List of hit dicts returned by `query_chroma`.

    Returns
    -------
    str
        A newline-separated string of labelled context snippets.
    """
    lines = [
        f"[{c['meta']['video_id']} {c['meta']['start']:.1f}-{c['meta']['end']:.1f}s] {c['text']}"
        for c in contexts
    ]
    return "\n\n".join(lines).strip()


def _answer_with_openai(question: str, contexts: list[dict]) -> str | None:
    """
    Call GPT-4o-mini to generate a grounded answer from the retrieved context.

    The model is instructed to answer strictly from the provided context and
    admit uncertainty rather than hallucinate.

    Parameters
    ----------
    question:
        The user's raw question string.
    contexts:
        Retrieved Chroma hits to use as grounding context.

    Returns
    -------
    str | None
        The model's answer, or None if:
        - OPENAI_API_KEY is not set.
        - langchain_openai is not installed.
        - The API call fails for any reason (quota, network, etc.).
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    ctx_text = _build_context(contexts)
    if not ctx_text:
        return None

    prompt = (
        "You are an expert analyst reviewing surveillance-video summaries.\n"
        "Answer the user's question using ONLY the context provided below.\n"
        "If the answer cannot be found in the context, say you don't know.\n\n"
        f"CONTEXT:\n{ctx_text}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER (concise, max 6 sentences):"
    )

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        return llm.invoke(prompt).content.strip()
    except Exception:
        return None


def _print_fallback(hits: list[dict]) -> None:
    """
    Display retrieved context snippets directly when the OpenAI call is unavailable.

    This local fallback ensures the demo remains useful even without an API key.
    """
    print("\nAssistant: OpenAI API unavailable (missing key, quota, or network error).")
    print("Here are the most relevant timeline snippets retrieved from the index:\n")
    for i, hit in enumerate(hits, start=1):
        m = hit["meta"]
        print(f"  {i}. {m['video_id']} [{m['start']:.1f}–{m['end']:.1f}s]  (distance={hit['distance']:.4f})")
        print(f"     {hit['text']}\n")


def interactive_chat(
    chroma_dir: Path,
    embedder_name: str,
    top_k: int = 6,
    collection_name: str = "video_summaries",
) -> None:
    """
    Start a blocking REPL for RAG-powered Q&A over video summaries.

    The loop continues until the user types "exit" or "quit".

    Parameters
    ----------
    chroma_dir:
        Path to the persisted ChromaDB index.
    embedder_name:
        SentenceTransformers model ID used during indexing.
    top_k:
        Number of context chunks to retrieve per query.
    collection_name:
        Chroma collection to query.
    """
    print("\n🎥  EcoSurve RAG Q&A  —  type 'exit' to quit\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break

        hits = query_chroma(
            chroma_dir=chroma_dir,
            embedder_name=embedder_name,
            query=question,
            top_k=top_k,
            collection_name=collection_name,
        )

        if not hits:
            print(
                "\nAssistant: Nothing was retrieved from the index. "
                "Have you run the pipeline and built the Chroma index?\n"
            )
            continue

        answer = _answer_with_openai(question, hits)
        if answer:
            print(f"\nAssistant: {answer}\n")
        else:
            _print_fallback(hits)


# ---------------------------------------------------------------------------
# Standalone entry-point (python src/rag/chat.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _chroma_dir = Path("outputs/chroma_db")
    _embedder = os.getenv("EMBEDDER_NAME", "all-MiniLM-L6-v2")
    interactive_chat(chroma_dir=_chroma_dir, embedder_name=_embedder)
"""
index.py — Build and query the Chroma vector store.

Summaries produced by the pipeline are embedded with SentenceTransformers and
stored in a persistent ChromaDB collection. Each video contributes:

- One "incident" document  — the video-level summary (spans the full duration).
- N "segment" documents    — one per 5-second window.

This granularity lets the RAG layer retrieve both broad context (incident) and
precise timeline snippets (segments) depending on the query.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Default collection name used across build and query calls.
_DEFAULT_COLLECTION = "video_summaries"


def build_chroma_from_summaries(
    summaries_dir: Path,
    chroma_dir: Path,
    embedder_name: str,
    collection_name: str = _DEFAULT_COLLECTION,
    reset: bool = True,
) -> dict:
    """
    Embed all summary JSON files and upsert them into a Chroma collection.

    The function accepts two JSON schemas produced by the pipeline:
    - Wrapped:   ``{ "<video_id>": { "sentences": [...], ... } }``
    - Flat:      ``{ "sentences": [...], ... }``

    Parameters
    ----------
    summaries_dir:
        Directory containing ``*.json`` summary files (one per video).
    chroma_dir:
        Path where ChromaDB will persist its on-disk index.
    embedder_name:
        SentenceTransformers model ID used for embedding.
    collection_name:
        Name of the Chroma collection to create / overwrite.
    reset:
        If True (default), delete the existing collection before indexing.
        This prevents duplicate documents on pipeline reruns.

    Returns
    -------
    dict
        A summary dict with keys ``collection``, ``count``, and ``chroma_dir``.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        # Drop the existing collection so reruns start from a clean slate.
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # Collection didn't exist yet — nothing to delete.

    collection = client.get_or_create_collection(name=collection_name)
    embedder = SentenceTransformer(embedder_name)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    embeddings: list[list[float]] = []

    for json_file in sorted(summaries_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        video_id = json_file.stem

        # Unwrap the outer video_id key if present (pipeline default format).
        if isinstance(data, dict) and "sentences" not in data and video_id in data:
            data = data[video_id]

        sentences: list[str] = data.get("sentences") or []
        timestamps: list[list[float]] = data.get("timestamps") or []
        incident: str = (data.get("incident_summary") or "").strip()
        duration: float = float(data.get("duration", 0))

        # --- Index the incident-level summary (full video span) ---
        if incident:
            ids.append(f"{video_id}::incident")
            docs.append(incident)
            metas.append({"video_id": video_id, "start": 0.0, "end": duration})
            embeddings.append(embedder.encode(incident).tolist())

        # --- Index each segment summary with its timestamp metadata ---
        for i, sentence in enumerate(sentences):
            text = (sentence or "").strip()
            if not text:
                continue

            start, end = 0.0, 0.0
            if i < len(timestamps) and len(timestamps[i]) == 2:
                start, end = float(timestamps[i][0]), float(timestamps[i][1])

            ids.append(f"{video_id}::seg::{i}")
            docs.append(text)
            metas.append({"video_id": video_id, "start": start, "end": end})
            embeddings.append(embedder.encode(text).tolist())

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    return {"collection": collection_name, "count": len(ids), "chroma_dir": str(chroma_dir)}


def query_chroma(
    chroma_dir: Path,
    embedder_name: str,
    query: str,
    top_k: int = 4,
    collection_name: str = _DEFAULT_COLLECTION,
) -> list[dict]:
    """
    Retrieve the *top_k* most semantically similar documents for a query.

    Parameters
    ----------
    chroma_dir:
        Path to the persisted ChromaDB index.
    embedder_name:
        SentenceTransformers model ID — must match the one used during indexing.
    query:
        Free-text question or search phrase.
    top_k:
        Number of results to return.
    collection_name:
        Chroma collection to query.

    Returns
    -------
    list[dict]
        Ranked list of hits, each containing:
        - ``text``     — the stored document string.
        - ``meta``     — dict with ``video_id``, ``start``, and ``end`` keys.
        - ``distance`` — cosine distance (lower = more similar).
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)
    embedder = SentenceTransformer(embedder_name)

    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return [
        {"text": doc, "meta": meta, "distance": float(dist)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
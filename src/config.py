"""
config.py — Centralised configuration for EcoSurve.

All tuneable knobs live here so nothing is hardcoded across the codebase.
Import `Paths` for filesystem layout and `PipelineConfig` for model/runtime settings.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """
    Canonical filesystem layout for the project.

    All paths are derived from the repo root so the project works regardless
    of where it is cloned on disk.
    """

    repo_root: Path = Path(__file__).resolve().parents[1]

    # Input
    videos_dir: Path = repo_root / "data" / "videos"

    # Outputs
    outputs_dir: Path = repo_root / "outputs"
    summaries_dir: Path = outputs_dir / "summaries"   # JSON + TXT summaries per video
    chroma_dir: Path = outputs_dir / "chroma_db"      # persistent vector store


@dataclass(frozen=True)
class PipelineConfig:
    """
    Runtime configuration for the VLM → Summarisation → RAG pipeline.

    Attributes
    ----------
    sample_fps:
        How many frames per second to sample from each video.
        Lower values are faster; higher values give more coverage.
        Default: 1 frame/s.
    segment_seconds:
        Duration (in seconds) of each summarisation window.
        Captions from all frames inside the window are collapsed into one sentence.
        Default: 5 s.
    max_frames_per_segment:
        Hard cap on frames processed per segment, regardless of `sample_fps`.
        Prevents runaway processing on high-FPS footage.
        Default: 8 frames.
    blip_model:
        HuggingFace model ID for the vision-language captioner.
        Supports both BLIP-1 ("Salesforce/blip-image-captioning-*")
        and BLIP-2 ("Salesforce/blip2-*") variants.
        Default: blip-image-captioning-large (fast, MPS-safe).
    summarizer_model:
        HuggingFace model ID for the seq2seq summariser (BART family).
        Used to compress per-segment captions and produce the incident summary.
        Default: facebook/bart-large-cnn.
    embedder_model:
        SentenceTransformers model ID used to embed text into the vector store.
        Default: all-MiniLM-L6-v2 (fast, strong retrieval quality).
    top_k:
        Number of nearest-neighbour chunks returned by Chroma per query.
        Default: 4.
    """

    # --- Frame sampling ---
    sample_fps: float = 1.0
    segment_seconds: float = 5.0
    max_frames_per_segment: int = 8

    # --- Models ---
    blip_model: str = "Salesforce/blip-image-captioning-large"
    summarizer_model: str = "facebook/bart-large-cnn"

    # --- RAG ---
    embedder_model: str = "all-MiniLM-L6-v2"
    top_k: int = 4
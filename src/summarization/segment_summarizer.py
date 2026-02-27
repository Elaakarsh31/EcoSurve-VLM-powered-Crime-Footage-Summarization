"""
segment_summarizer.py — Compresses per-frame captions into concise summaries.

Uses a local HuggingFace seq2seq model (default: facebook/bart-large-cnn) so
no API key is required for this stage. The summariser is used at two levels:

1. Segment level  — collapses 4–8 captions from a 5-second window into one sentence.
2. Incident level — collapses all segment summaries into a single video-level summary.

Long inputs are automatically chunked to stay within the model's token budget,
then the chunk summaries are merged and optionally re-compressed.
"""

from __future__ import annotations

from transformers import pipeline


class SegmentSummarizer:
    """
    Compress a list of text captions into a short, coherent summary.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for a seq2seq summarisation model.
        Tested with "facebook/bart-large-cnn".
    device:
        Torch device to run the model on. Accepts None (auto), "mps",
        a CUDA device index (int ≥ 0), or -1 for CPU.
    """

    def __init__(self, model_name: str, device: str | int | None = None) -> None:
        kwargs = {}
        if device is not None:
            kwargs["device"] = device

        self._pipe = pipeline(
            task="summarization",
            model=model_name,
            tokenizer=model_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_join(captions: list[str]) -> str:
        """Strip whitespace from each caption and join with a single space."""
        return " ".join(c.strip() for c in captions if c and c.strip())

    @staticmethod
    def _pick_lengths(word_count: int) -> tuple[int, int]:
        """
        Compute (max_length, min_length) token targets for the summariser.

        The heuristic targets ~35 % of the input word count, bounded to
        sensible absolute limits so we never ask BART to generate more tokens
        than it received (which raises an error).

        Returns (0, 0) for very short inputs to signal "skip summarisation".
        """
        if word_count < 25:
            # Input is already short enough — no summarisation needed.
            return 0, 0

        target = max(20, int(word_count * 0.35))
        max_len = min(120, max(35, target))
        min_len = min(max(15, int(max_len * 0.6)), max_len - 5)
        return max_len, min_len

    def _run_summarizer(self, text: str, max_length: int, min_length: int) -> str:
        """
        Call the underlying HuggingFace pipeline on a single text block.

        `truncation=True` is set to prevent errors when the tokenised input
        exceeds the model's maximum context length.
        """
        result = self._pipe(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"].strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_captions(self, captions: list[str]) -> str:
        """
        Summarise a list of captions into a single sentence or short paragraph.

        The method handles three cases transparently:

        1. Empty input  → returns a safe placeholder string.
        2. Short input  → returns the joined captions as-is (no BART call).
        3. Long input   → splits into ~220-word chunks, summarises each chunk
                          independently, then optionally merges the results
                          with a final summarisation pass.

        Parameters
        ----------
        captions:
            A list of natural-language strings (typically BLIP frame captions
            or previously generated segment summaries).

        Returns
        -------
        str
            A compressed, human-readable summary of the input captions.
        """
        text = self._clean_join(captions)
        if not text:
            return "No visible activity detected in this segment."

        words = text.split()
        max_len, min_len = self._pick_lengths(len(words))

        # Input is short enough to return directly without summarisation.
        if max_len == 0:
            return text

        # --- Single-pass summarisation for inputs within the chunk budget ---
        chunk_size = 220  # words; stays safely within BART's 1 024-token limit
        if len(words) <= chunk_size:
            return self._run_summarizer(text, max_len, min_len)

        # --- Multi-chunk summarisation for long inputs ---
        # Summarise each chunk independently, then merge.
        chunk_summaries: list[str] = []
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i: i + chunk_size])
            c_max, c_min = self._pick_lengths(len(chunk_text.split()))
            if c_max == 0:
                chunk_summaries.append(chunk_text)
            else:
                chunk_summaries.append(self._run_summarizer(chunk_text, c_max, c_min))

        # Merge chunk summaries and optionally compress again.
        merged = " ".join(chunk_summaries).strip()
        m_max, m_min = self._pick_lengths(len(merged.split()))
        if m_max == 0:
            return merged

        return self._run_summarizer(merged, m_max, m_min)
"""
io.py — Filesystem utility helpers.

Provides thin wrappers around common file operations so the rest of the
codebase avoids repetitive boilerplate and stays easy to test.
"""

from __future__ import annotations

import json
from pathlib import Path

# Video file extensions recognised by the pipeline.
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv"})


def list_videos(videos_dir: Path) -> list[Path]:
    """
    Return a sorted list of video files found directly inside *videos_dir*.

    Only files whose extension matches `VIDEO_EXTS` are included.
    Subdirectories and non-video files are silently skipped.

    Parameters
    ----------
    videos_dir:
        Directory to search (non-recursive).

    Returns
    -------
    list[Path]
        Alphabetically sorted list of video file paths.
        Returns an empty list if the directory does not exist.
    """
    if not videos_dir.exists():
        return []
    return sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def ensure_dir(path: Path) -> None:
    """
    Create *path* and all intermediate parents if they do not already exist.

    Equivalent to ``mkdir -p``. Safe to call on an existing directory.
    """
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object, indent: int = 2) -> None:
    """
    Serialise *data* to a JSON file at *path*, overwriting if it exists.

    Parameters
    ----------
    path:
        Destination file path (parent directory must exist).
    data:
        Any JSON-serialisable Python object.
    indent:
        Pretty-print indentation level. Default: 2.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def write_text(path: Path, text: str) -> None:
    """
    Write *text* to a UTF-8 encoded file at *path*, overwriting if it exists.

    Parameters
    ----------
    path:
        Destination file path (parent directory must exist).
    text:
        String content to write.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
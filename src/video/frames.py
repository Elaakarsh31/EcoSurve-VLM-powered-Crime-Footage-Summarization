"""
frames.py — Video reading and frame-sampling utilities.

Provides two public helpers:
- `read_video_meta`     — cheaply inspect a video file's properties.
- `iter_sampled_frames` — lazily yield frames at a configurable rate,
                          optionally constrained to a time window.
"""

from __future__ import annotations

import cv2
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class VideoMeta:
    """
    Lightweight container for video file properties.

    Attributes
    ----------
    fps:         Native frame rate reported by the container.
    frame_count: Total number of frames in the video.
    duration_s:  Computed duration in seconds (frame_count / fps).
    width:       Frame width in pixels.
    height:      Frame height in pixels.
    """

    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


def read_video_meta(video_path: str) -> VideoMeta:
    """
    Open a video file, read its metadata, and close it immediately.

    This is intentionally lightweight — it does not decode any frames.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the video file.

    Returns
    -------
    VideoMeta
        Struct containing FPS, frame count, duration, and resolution.

    Raises
    ------
    RuntimeError
        If OpenCV cannot open the file (missing codec, corrupt file, etc.).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_s = (frame_count / fps) if fps > 0 else 0.0
    return VideoMeta(fps=fps, frame_count=frame_count, duration_s=duration_s,
                     width=width, height=height)


def iter_sampled_frames(
    video_path: str,
    sample_fps: float,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Iterator[tuple[float, "cv2.Mat"]]:
    """
    Lazily yield frames sampled at a target rate from a time window.

    Rather than decoding every frame, the function seeks directly to the
    required frame indices, minimising CPU/memory overhead for long videos.

    Parameters
    ----------
    video_path:
        Path to the video file.
    sample_fps:
        Desired output frame rate (e.g. 1.0 → one frame per second).
        The actual rate is quantised to the nearest integer step of the
        native FPS, so the true rate may differ slightly.
    start_s:
        Start of the sampling window in seconds. Default: 0.
    end_s:
        End of the sampling window in seconds.
        If None, the window extends to the end of the video.
    max_frames:
        If set, stop after yielding this many frames (hard cap).
        Useful for limiting processing per segment.

    Yields
    ------
    tuple[float, cv2.Mat]
        A (timestamp_seconds, frame_bgr) pair for each sampled frame.

    Raises
    ------
    RuntimeError
        If OpenCV cannot open the file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if native_fps <= 0:
        native_fps = 30.0  # safe fallback for malformed containers

    # Resolve end_s against actual video length if not provided
    if end_s is None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        end_s = (frame_count / native_fps) if frame_count > 0 else start_s

    # Convert time bounds to frame indices
    start_frame = int(max(0.0, start_s) * native_fps)
    end_frame = int(max(0.0, end_s) * native_fps)

    # Number of native frames to skip between each sampled frame
    step = max(1, int(round(native_fps / max(sample_fps, 0.0001))))

    yielded = 0
    for frame_idx in range(start_frame, end_frame, step):
        # Seek directly to the target frame — avoids decoding skipped frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        yield (frame_idx / native_fps, frame)

        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            break

    cap.release()
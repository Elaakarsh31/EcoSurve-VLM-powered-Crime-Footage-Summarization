"""
blip_captioner.py — Vision-language frame captioner.

Wraps both BLIP-1 and BLIP-2 behind a single `BlipCaptioner` interface so the
rest of the pipeline never needs to care which model family is loaded.

Supported models
----------------
- "Salesforce/blip-image-captioning-large"  — fast, MPS/CPU-safe (recommended for local demo)
- "Salesforce/blip2-opt-2.7b"               — higher quality, requires ~6 GB VRAM

Device handling
---------------
- CUDA  → float16  (fast, memory-efficient)
- MPS   → float32  (float16 causes silent hangs on Apple Silicon)
- CPU   → float32
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

# Model name prefixes that require the BLIP-2 class pair instead of BLIP-1.
_BLIP2_PREFIXES = ("Salesforce/blip2",)


def _is_blip2(model_name: str) -> bool:
    """Return True if *model_name* belongs to the BLIP-2 family."""
    return any(model_name.startswith(p) for p in _BLIP2_PREFIXES)


def _inference_dtype(device: str) -> torch.dtype:
    """
    Choose the safest dtype for inference on the given device.

    CUDA supports float16 without issue.
    MPS (Apple Silicon) and CPU must use float32 — float16 on MPS causes
    model.generate() to hang silently with no error.
    """
    return torch.float16 if device == "cuda" else torch.float32


class BlipCaptioner:
    """
    Generate natural-language captions from BGR video frames.

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Both BLIP-1 and BLIP-2 variants are supported.
    device:
        Torch device string ("cuda", "mps", "cpu").
        Auto-detected from hardware if not supplied.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _inference_dtype(self.device)

        if _is_blip2(model_name):
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype
            ).to(self.device)
        else:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(
                self.device
            )

        self.model.eval()

    @torch.inference_mode()
    def caption_bgr(self, frame_bgr: np.ndarray, max_new_tokens: int = 50) -> str:
        """
        Generate a caption for a single BGR video frame.

        OpenCV yields frames in BGR order; this method converts to RGB before
        passing the image to the processor.

        Parameters
        ----------
        frame_bgr:
            A uint8 numpy array of shape (H, W, 3) in BGR channel order.
        max_new_tokens:
            Maximum number of tokens the model may generate.

        Returns
        -------
        str
            A natural-language description of the frame's content.
        """
        # OpenCV BGR → RGB → PIL Image (required by HuggingFace processors)
        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb.astype(np.uint8))

        if _is_blip2(self.model_name):
            return self._caption_blip2(image, max_new_tokens)
        else:
            return self._caption_blip1(image, max_new_tokens)

    def _caption_blip2(self, image: Image.Image, max_new_tokens: int) -> str:
        """
        Caption using BLIP-2 with a task-specific prompt.

        BLIP-2 supports visual question answering via a text prompt, which
        significantly improves relevance for surveillance footage compared to
        unconditional captioning.
        """
        dtype = _inference_dtype(self.device)
        prompt = "Question: Describe what is happening in this surveillance footage. Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, dtype
        )
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # The decoded output includes the full prompt — keep only the answer part.
        full_text = self.processor.decode(out[0], skip_special_tokens=True)
        answer = full_text.split("Answer:")[-1].strip()
        return answer if answer else full_text.strip()

    def _caption_blip1(self, image: Image.Image, max_new_tokens: int) -> str:
        """Caption using BLIP-1 (unconditional generation)."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()
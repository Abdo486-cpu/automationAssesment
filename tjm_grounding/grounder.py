"""
grounder.py — VLM visual-grounding backends.

Defines the `Grounder` Protocol so the rest of the codebase depends ONLY on the
interface, never on a specific model or API.  Callers pass a PIL Image and a
natural-language query; they get back a (x, y, confidence) triple or None.

No template matching.  No reference images.  The model is the oracle.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol, runtime_checkable

from PIL import Image

from tjm_grounding import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value object returned by every grounder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundingResult:
    """
    x, y  — pixel coordinates in the coordinate space of the image that was
             passed to locate().  Callers must re-project to full-screen when
             the image is a crop.
    confidence — [0, 1].  Below config.CONFIDENCE_THRESHOLD the caller should
                 widen the search and retry.
    raw   — the raw model response string, kept for logging/debugging.
    """
    x: int
    y: int
    confidence: float
    raw: str = ""


# ---------------------------------------------------------------------------
# Protocol — the only thing the rest of the codebase imports
# ---------------------------------------------------------------------------

@runtime_checkable
class Grounder(Protocol):
    def locate(self, image: Image.Image, query: str) -> GroundingResult | None:
        """
        Ground *query* in *image*.  Return None if the model cannot locate it
        (e.g. element absent, model timed-out, parse failure after retries).
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data-URL string."""
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_point(text: str, img_w: int, img_h: int) -> tuple[int, int] | None:
    """
    Parse a model response that contains coordinates.

    Supported formats (in order of preference):
      1. JSON with keys x/y (integers or 0-1 floats):
            {"x": 0.34, "y": 0.72}  or  {"x": 312, "y": 740}
      2. JSON with a "point" list:  [0.34, 0.72]
      3. JSON bounding box "bbox": [x0,y0,x1,y1] (any of abs or norm) — center used
      4. Bare "[x, y]" or "(x, y)" anywhere in the text
    Returns pixel coords in the image's own coordinate space.
    """
    text = text.strip()

    # 1 / 2 / 3 — try JSON first
    json_match = re.search(r"\{[^}]+\}", text, re.S)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            if "x" in obj and "y" in obj:
                return _norm_or_abs(obj["x"], obj["y"], img_w, img_h)
            if "point" in obj:
                pt = obj["point"]
                return _norm_or_abs(pt[0], pt[1], img_w, img_h)
            if "bbox" in obj:
                b = obj["bbox"]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                return _norm_or_abs(cx, cy, img_w, img_h)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass

    # 4 — bare list anywhere in text
    list_match = re.search(r"[\[\(]\s*([0-9.]+)\s*,\s*([0-9.]+)\s*[\]\)]", text)
    if list_match:
        return _norm_or_abs(float(list_match.group(1)), float(list_match.group(2)), img_w, img_h)

    return None


def _norm_or_abs(vx: float, vy: float, w: int, h: int) -> tuple[int, int]:
    """
    If both values are in [0, 1] treat them as normalised fractions; otherwise
    treat as absolute pixels.  The heuristic works for all Qwen2.5-VL outputs.
    """
    if 0.0 <= vx <= 1.0 and 0.0 <= vy <= 1.0:
        return int(vx * w), int(vy * h)
    return int(vx), int(vy)


def _extract_confidence(text: str) -> float:
    """
    Pull a confidence score from free-form model text.  Looks for patterns
    like 'confidence: 0.92', '"confidence": 0.9', or 'confidence=0.9'.
    Defaults to 0.8 when the model doesn't report one (assume success if coords parsed).
    """
    # Allow an optional quote between the key name and the colon/equals
    # and an optional minus sign (for clamping tests).
    m = re.search(r"confidence[\"']?\s*[:=]\s*(-?[0-9.]+)", text, re.I)
    if m:
        try:
            return min(1.0, max(0.0, float(m.group(1))))
        except ValueError:
            pass
    return 0.8  # optimistic default when coords parsed OK


# ---------------------------------------------------------------------------
# Backend 1 — OpenRouter / Qwen2.5-VL (default, used in production)
# ---------------------------------------------------------------------------

class ApiVlmGrounder:
    """
    Calls a VLM through OpenRouter using the OpenAI-compatible chat endpoint.

    The prompt asks the model to return *only* a JSON object with the pixel
    (or 0-1 normalised) coordinates of the described element.  This keeps the
    response small and parseable; free-form prose is tolerated as a fallback.
    """

    SYSTEM_PROMPT = (
        "You are a GUI element locator.  "
        "Given a screenshot and a description of a UI element, return ONLY a "
        "JSON object: {\"x\": <0-1 fraction>, \"y\": <0-1 fraction>, "
        "\"confidence\": <0-1>}.  "
        "x=0,y=0 is top-left.  If the element is absent return {\"x\": null}."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._model = model or config.VLM_MODEL
        self._client = OpenAI(
            api_key=api_key or config.OPENROUTER_API_KEY,
            base_url=base_url or config.OPENROUTER_BASE_URL,
        )

    def locate(self, image: Image.Image, query: str) -> GroundingResult | None:
        w, h = image.size
        b64 = _pil_to_b64(image)
        prompt = (
            f"Locate the following element in this screenshot and return its "
            f"centre as a JSON {{x, y}} object with values in [0, 1]:\n\n"
            f"Element: {query}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=128,
                temperature=0.0,
            )
        except Exception as exc:
            log.error("VLM API call failed: %s", exc)
            return None

        raw = response.choices[0].message.content or ""
        log.debug("VLM raw response: %s", raw)

        # Model says element is absent
        if '"x": null' in raw or "not found" in raw.lower():
            log.info("VLM reports element absent for query: %r", query)
            return None

        point = _extract_point(raw, w, h)
        if point is None:
            log.warning("Could not parse VLM response: %s", raw)
            return None

        x, y = point
        conf = _extract_confidence(raw)
        return GroundingResult(x=x, y=y, confidence=conf, raw=raw)

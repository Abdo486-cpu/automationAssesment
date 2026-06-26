"""
narrowing.py — ScreenSpot-Pro coarse→fine grounding loop.

Why coarse→fine?
  High-res desktop screenshots overwhelm a VLM's effective receptive field;
  small icons become sub-pixel blobs.  By first locating an approximate region,
  then re-querying a zoomed crop, we trade two cheaper calls for one expensive
  (and often inaccurate) full-image call.  This matches the ScreenSpot-Pro
  paper's finding that cropping+upscaling is the single biggest accuracy lever
  on high-DPI screens.

Public API: `ground(image, query, grounder) -> GroundingResult | None`
"""
from __future__ import annotations

import logging
import math

from PIL import Image

from tjm_grounding import config
from tjm_grounding.grounder import Grounder, GroundingResult

log = logging.getLogger(__name__)

def scale_point(
    px: int, py: int,
    src_w: int, src_h: int,
    dst_w: int, dst_h: int,
) -> tuple[int, int]:
    """
    Re-project a point from one image space to another.

    Used twice:
      • coarse pass:  model coords in downscaled image → full-res coords
      • fine pass:    model coords inside the view of the crop → full-res coords
    """
    x_out = int(px * dst_w / src_w)
    y_out = int(py * dst_h / src_h)
    return x_out, y_out


def crop_coords(
    cx: int, cy: int,
    crop_half: int,
    img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """
    Return (x0, y0, x1, y1) for a square crop centred on (cx, cy), clamped to
    the image boundaries.
    """
    x0 = max(0, cx - crop_half)
    y0 = max(0, cy - crop_half)
    x1 = min(img_w, cx + crop_half)
    y1 = min(img_h, cy + crop_half)
    return x0, y0, x1, y1


def reproject_crop_to_full(
    px: int, py: int,
    crop_x0: int, crop_y0: int,
    crop_w: int, crop_h: int,
    view_w: int, view_h: int,
) -> tuple[int, int]:
    """
    Convert a point (px, py) expressed in the *displayed* view of a crop back
    to full-screen coordinates.

    The fine-pass image is the crop up-scaled from (crop_w × crop_h) to
    (view_w × view_h).  The VLM returns coords in view space.  We must:
      1. Scale back to crop space:   x_crop = px * crop_w / view_w
      2. Shift by crop origin:       x_full = x_crop + crop_x0
    """
    x_full = int(px * crop_w / view_w) + crop_x0
    y_full = int(py * crop_h / view_h) + crop_y0
    return x_full, y_full


def downscale_image(image: Image.Image, long_edge: int) -> Image.Image:
    """Resize *image* so its longer dimension equals *long_edge* (aspect preserved)."""
    w, h = image.size
    scale = long_edge / max(w, h)
    if scale >= 1.0:
        return image
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Fast single-pass check (coarse only) — used for popup sweeps where speed
# matters more than precision.
# ---------------------------------------------------------------------------

def quick_check(
    full_image: Image.Image,
    query: str,
    grounder: Grounder,
    *,
    coarse_long_edge: int = config.COARSE_LONG_EDGE,
) -> GroundingResult | None:
    """One coarse API call only — fast enough for popup detection."""
    coarse_img = downscale_image(full_image, coarse_long_edge)
    coarse_w, coarse_h = coarse_img.size
    result = grounder.locate(coarse_img, query)
    if result is None:
        return None
    full_w, full_h = full_image.size
    x, y = scale_point(result.x, result.y, coarse_w, coarse_h, full_w, full_h)
    return GroundingResult(x=x, y=y, confidence=result.confidence, raw=result.raw)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ground(
    full_image: Image.Image,
    query: str,
    grounder: Grounder,
    *,
    coarse_long_edge: int = config.COARSE_LONG_EDGE,
    crop_size_px: int = config.CROP_SIZE_PX,
    fine_view_px: int = config.FINE_VIEW_PX,
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    max_retries: int = config.MAX_GROUNDING_RETRIES,
    crop_widen_factor: float = config.CROP_WIDEN_FACTOR,
) -> GroundingResult | None:
    """
    Coarse→fine visual grounding.  Returns a GroundingResult in *full_image*
    pixel space, or None if all retries fail.
    """
    full_w, full_h = full_image.size

    # ------------------------------------------------------------------
    # COARSE PASS: downscale the whole desktop, ask for a rough region.
    # ------------------------------------------------------------------
    coarse_img = downscale_image(full_image, coarse_long_edge)
    coarse_w, coarse_h = coarse_img.size
    log.info("[coarse] querying %dx%d image for: %r", coarse_w, coarse_h, query)

    coarse_result = grounder.locate(coarse_img, query)
    if coarse_result is None:
        log.warning("[coarse] model could not locate %r", query)
        return None

    # Re-project coarse coords to full-screen space.
    cx_full, cy_full = scale_point(
        coarse_result.x, coarse_result.y,
        coarse_w, coarse_h,
        full_w, full_h,
    )
    log.info("[coarse] estimate at full-res: (%d, %d)", cx_full, cy_full)

    current_crop_size = crop_size_px

    for attempt in range(max_retries):
        # ------------------------------------------------------------------
        # FINE PASS: crop around the coarse estimate and up-scale it.
        # ------------------------------------------------------------------
        half = current_crop_size // 2
        x0, y0, x1, y1 = crop_coords(cx_full, cy_full, half, full_w, full_h)
        crop_w = x1 - x0
        crop_h = y1 - y0

        crop = full_image.crop((x0, y0, x1, y1))
        # Up-scale: raise effective pixel density so small icons are legible.
        fine_img = crop.resize((fine_view_px, fine_view_px), Image.LANCZOS)

        log.info(
            "[fine attempt %d] crop (%d,%d)–(%d,%d) → %dx%d view",
            attempt, x0, y0, x1, y1, fine_view_px, fine_view_px,
        )

        fine_result = grounder.locate(fine_img, query)
        if fine_result is None:
            log.warning("[fine] model could not locate %r in crop, widening", query)
            current_crop_size = int(current_crop_size * crop_widen_factor)
            continue

        # Re-project fine coords back to full-screen.
        x_full, y_full = reproject_crop_to_full(
            fine_result.x, fine_result.y,
            x0, y0,
            crop_w, crop_h,
            fine_view_px, fine_view_px,
        )
        log.info("[fine] projected to full-res: (%d, %d)", x_full, y_full)

        # Trust the fine result if confidence is acceptable; skip the extra
        # verify API call to keep the total down to 2 calls per grounding.
        if fine_result.confidence >= confidence_threshold:
            return GroundingResult(
                x=x_full,
                y=y_full,
                confidence=fine_result.confidence,
                raw=fine_result.raw,
            )

        log.warning(
            "[fine] confidence %.2f below threshold %.2f on attempt %d, widening",
            fine_result.confidence, confidence_threshold, attempt,
        )
        current_crop_size = int(current_crop_size * crop_widen_factor)
        cx_full, cy_full = x_full, y_full

    log.error(
        "Grounding failed after %d attempts for query: %r", max_retries, query
    )
    return None

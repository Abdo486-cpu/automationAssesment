from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tjm_grounding.grounder import GroundingResult


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _load_font(size: int = 18) -> ImageFont.ImageFont:
    """Return a truetype or fallback bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_grounding(
    image: Image.Image,
    result: GroundingResult,
    query: str,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    color: str = "#FF3333",
    radius: int = 12,
) -> Image.Image:
    """
    Return an annotated copy of *image* with:
      • A circle at the final click point (result.x, result.y)
      • Optional rectangle showing the crop window used
      • A label with the query and confidence score
    """
    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(16)

    # Crop rectangle (semi-transparent yellow)
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 200, 0, 200), width=3)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 200, 0, 30))

    # Click-point circle
    cx, cy = result.x, result.y
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(255, 51, 51, 180),
        outline=(255, 255, 255, 230),
        width=3,
    )
    # Cross-hair lines
    draw.line([cx - radius * 2, cy, cx + radius * 2, cy], fill=(255, 255, 255, 200), width=2)
    draw.line([cx, cy - radius * 2, cx, cy + radius * 2], fill=(255, 255, 255, 200), width=2)

    # Label
    label = f"{query[:50]}  [{result.confidence:.0%}]"
    padding = 4
    bbox = font.getbbox(label)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    label_x = max(0, cx - text_w // 2)
    label_y = max(0, cy - radius - text_h - padding * 2 - 4)
    draw.rectangle(
        [label_x - padding, label_y - padding,
         label_x + text_w + padding, label_y + text_h + padding],
        fill=(0, 0, 0, 160),
    )
    draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(annotated, overlay).convert("RGB")


def save_annotated(
    image: Image.Image,
    result: GroundingResult,
    query: str,
    path: Path,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Path:
    """Annotate and save to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_grounding(image, result, query, crop_box=crop_box)
    annotated.save(str(path))
    return path

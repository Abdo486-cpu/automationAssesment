"""Central configuration — all magic numbers live here, nowhere else."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# VLM / OpenRouter
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Qwen2.5-VL is the default because it consistently outputs normalized
# coordinates (0-1) and handles high-res crops without hallucinating.
VLM_MODEL: str = os.environ.get("VLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")

# ---------------------------------------------------------------------------
# Narrowing / coarse-to-fine
# ---------------------------------------------------------------------------
# Full-desktop downscale: send this many pixels on the long edge to the VLM.
# Keeps tokens/latency low; the model only needs to locate a rough region.
COARSE_LONG_EDGE: int = int(os.environ.get("COARSE_LONG_EDGE", "1024"))

# After the coarse pass, cut a crop this wide/tall around the estimated point.
CROP_SIZE_PX: int = int(os.environ.get("CROP_SIZE_PX", "400"))

# The crop is up-scaled to this for the fine pass (raises effective resolution).
FINE_VIEW_PX: int = int(os.environ.get("FINE_VIEW_PX", "800"))

# If fine-pass confidence < this, widen the crop and retry.
CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

# Maximum widen-and-retry iterations before aborting a grounding attempt.
MAX_GROUNDING_RETRIES: int = int(os.environ.get("MAX_GROUNDING_RETRIES", "3"))

# Multiplier applied to CROP_SIZE_PX each retry.
CROP_WIDEN_FACTOR: float = float(os.environ.get("CROP_WIDEN_FACTOR", "1.5"))

# ---------------------------------------------------------------------------
# Pop-up sweep
# ---------------------------------------------------------------------------
POPUP_QUERY: str = "any modal dialog, alert, or pop-up window and its dismiss/close/OK button"

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
POSTS_URL: str = "https://jsonplaceholder.typicode.com/posts"
MAX_POSTS: int = 10

NOTEPAD_QUERY: str = "the center of the Notepad application shortcut icon on the desktop, not the shortcut arrow"
NOTEPAD_WINDOW_TITLE: str = "Notepad"  # substring match

# How many times to retry the grounding+click before giving up on a post.
CLICK_RETRY_LIMIT: int = int(os.environ.get("CLICK_RETRY_LIMIT", "3"))

# Seconds to poll for the Notepad window after double-clicking.
WINDOW_POLL_TIMEOUT: float = float(os.environ.get("WINDOW_POLL_TIMEOUT", "5.0"))
WINDOW_POLL_INTERVAL: float = 0.5

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = Path.home() / "Desktop" / "tjm-project"
SCREENSHOT_DIR: Path = OUTPUT_DIR / "_screenshots"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

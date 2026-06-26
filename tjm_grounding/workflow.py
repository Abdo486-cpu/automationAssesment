from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from tjm_grounding import actuation, annotate, config, narrowing
from tjm_grounding.grounder import Grounder, GroundingResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class Post:
    id: int
    title: str
    body: str


class WorkflowError(RuntimeError):
    """Raised when a post cannot be processed; wraps context for structured logging."""

    def __init__(self, message: str, *, post_id: int, stage: str, screenshot_path: Path | None = None):
        super().__init__(message)
        self.post_id = post_id
        self.stage = stage
        self.screenshot_path = screenshot_path


# ---------------------------------------------------------------------------
# Data helpers  (unit-tested)
# ---------------------------------------------------------------------------

def fetch_posts(url: str = config.POSTS_URL, limit: int = config.MAX_POSTS) -> list[Post]:
    """Fetch posts from JSONPlaceholder and return the first *limit*."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    raw: list[dict[str, Any]] = resp.json()
    return [Post(id=p["id"], title=p["title"], body=p["body"]) for p in raw[:limit]]


def format_post(post: Post) -> str:
    """Format a post for typing into Notepad."""
    return f"Title: {post.title}\n\n{post.body}"


def post_filename(post: Post) -> str:
    """Return the deterministic filename for a post."""
    return f"post_{post.id}.txt"


def post_filepath(post: Post) -> Path:
    return config.OUTPUT_DIR / post_filename(post)


# ---------------------------------------------------------------------------
# Desktop clearing — minimize-all + popup sweep until the desktop is clean
# ---------------------------------------------------------------------------

MAX_CLEAR_ATTEMPTS = 5

def _clear_desktop(grounder: Grounder) -> Image.Image:
    """
    Minimize all windows, screenshot, and dismiss any popup the VLM spots;
    repeat until clean or MAX_CLEAR_ATTEMPTS.  Returns the final screenshot.
    """
    for attempt in range(MAX_CLEAR_ATTEMPTS):
        actuation.show_desktop()
        screenshot = actuation.capture_screenshot()

        result = narrowing.quick_check(screenshot, config.POPUP_QUERY, grounder)
        if result is None:
            log.info("Desktop clear (attempt %d)", attempt + 1)
            return screenshot

        log.warning(
            "Pop-up detected on attempt %d — dismissing at (%d, %d)",
            attempt + 1, result.x, result.y,
        )
        actuation.click(result.x, result.y)
        time.sleep(0.3)  # let the popup animate away before re-checking

    # After max attempts just return whatever we have and let grounding decide
    log.warning("Could not fully clear desktop after %d attempts, proceeding anyway", MAX_CLEAR_ATTEMPTS)
    return actuation.capture_screenshot()


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(grounder: Grounder) -> None:
    """
    Full run: fetch 10 posts, for each post open Notepad, type, save, close.
    Continues to the next post if any step fails (logs the error with context).
    """
    actuation.ensure_dir(config.OUTPUT_DIR)
    actuation.ensure_dir(config.SCREENSHOT_DIR)

    posts = fetch_posts()
    log.info("Fetched %d posts", len(posts))

    for post in posts:
        try:
            _process_post(post, grounder)
        except WorkflowError as exc:
            log.error(
                "Post %d failed at stage '%s': %s  (screenshot: %s)",
                exc.post_id, exc.stage, exc, exc.screenshot_path,
            )


def _process_post(post: Post, grounder: Grounder) -> None:
    log.info("=== Processing post %d ===", post.id)

    # ------------------------------------------------------------------
    # 1. Clear the desktop: Win+D loop until no popups remain.
    #    Each retry presses Win+D again so anything that re-appeared is caught.
    # ------------------------------------------------------------------
    screenshot = _clear_desktop(grounder)

    # ------------------------------------------------------------------
    # 2. Ground the Notepad icon; re-clear on each retry in case a popup
    #    appeared between attempts.
    # ------------------------------------------------------------------
    grounding_result: GroundingResult | None = None
    for attempt in range(config.CLICK_RETRY_LIMIT):
        grounding_result = narrowing.ground(screenshot, config.NOTEPAD_QUERY, grounder)
        if grounding_result is not None:
            break
        log.warning("Icon grounding attempt %d/%d failed, re-clearing desktop", attempt + 1, config.CLICK_RETRY_LIMIT)
        time.sleep(1.0)
        screenshot = _clear_desktop(grounder)

    if grounding_result is None:
        raise WorkflowError(
            "Could not ground Notepad icon after all retries",
            post_id=post.id, stage="GROUND_ICON",
        )

    # Save annotated deliverable screenshot
    ann_path = config.SCREENSHOT_DIR / f"post_{post.id:02d}_1_grounded.png"
    annotate.save_annotated(screenshot, grounding_result, config.NOTEPAD_QUERY, ann_path)
    log.info("Annotated grounding saved: %s", ann_path)

    actuation.show_desktop()  # minimize everything so the icon is visible

    opened = False
    for launch_attempt in range(config.CLICK_RETRY_LIMIT):
        if not actuation.focus_desktop():
            left, top, right, bottom = actuation.get_work_area()
            empty_x, empty_y = right - 40, (top + bottom) // 2
            actuation.click(empty_x, empty_y)
            time.sleep(0.2)
            actuation.focus_desktop()  # best-effort; proceed regardless

        actuation.double_click(grounding_result.x, grounding_result.y)

        opened = actuation.wait_for_window(
            config.NOTEPAD_WINDOW_TITLE,
            timeout=config.WINDOW_POLL_TIMEOUT,
            interval=config.WINDOW_POLL_INTERVAL,
        )
        if opened:
            break
        log.warning(
            "Notepad did not appear (launch attempt %d/%d), retrying",
            launch_attempt + 1, config.CLICK_RETRY_LIMIT,
        )

    if not opened:
        raise WorkflowError(
            f"Notepad window did not appear after {config.CLICK_RETRY_LIMIT} launch attempts",
            post_id=post.id, stage="CONFIRM_WINDOW", screenshot_path=ann_path,
        )
    time.sleep(0.3)  # let Notepad fully render

    # 5. Ctrl+N for a fresh blank tab
    actuation.hotkey("ctrl", "n")
    time.sleep(0.5)  # wait for the new blank window to take focus
    content = format_post(post)
    actuation.paste_text(content)

    # 6. Save via Ctrl+S → type file path → Enter.
    save_path = post_filepath(post)
    actuation.hotkey("ctrl", "s")
    time.sleep(0.5)  # wait for Save-As dialog
    actuation.paste_text(str(save_path))
    time.sleep(0.3)
    actuation.press("enter")
    time.sleep(0.3)

    # If Windows shows an overwrite/confirm prompt, dismiss it (same loop logic).
    for _ in range(3):
        shot = actuation.capture_screenshot()
        result = narrowing.quick_check(shot, config.POPUP_QUERY, grounder)
        if result is None:
            break
        log.warning("Overwrite prompt detected — dismissing at (%d, %d)", result.x, result.y)
        actuation.click(result.x, result.y)
        time.sleep(0.3)

    # 7. Close Notepad, then wait until it's truly gone
    closed = actuation.close_window(config.NOTEPAD_WINDOW_TITLE)
    if not closed:
        log.warning("Post %d: could not close Notepad window, continuing", post.id)

    gone = actuation.wait_for_window_gone(
        config.NOTEPAD_WINDOW_TITLE,
        timeout=config.WINDOW_POLL_TIMEOUT,
        interval=config.WINDOW_POLL_INTERVAL,
    )
    if not gone:
        log.warning("Post %d: Notepad window still present after close timeout", post.id)

    log.info("Post %d complete → %s", post.id, save_path)

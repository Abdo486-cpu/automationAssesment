"""
actuation.py — screen capture and mouse/keyboard input.

This layer NEVER decides where to click; it only captures screenshots (mss),
clicks/types (pyautogui), and queries windows (pygetwindow).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# Drop pyautogui's default 0.1s pause between actions (it stacks up); keep the
# failsafe (slam the mouse to (0,0) to kill the script).
try:
    import pyautogui as _pag
    _pag.PAUSE = 0
    _pag.FAILSAFE = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def capture_screenshot() -> Image.Image:
    """Capture the full primary monitor as a PIL RGB image."""
    import mss  # local import: unavailable on Linux CI

    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[1])  # index 1 = primary monitor
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


# ---------------------------------------------------------------------------
# Mouse & keyboard
# ---------------------------------------------------------------------------

def double_click(x: int, y: int) -> None:
    """Double-click at (x, y).  interval=0.1 so the two clicks register as a pair."""
    import pyautogui

    log.debug("double_click(%d, %d)", x, y)
    pyautogui.doubleClick(x, y, interval=0.1)


def click(x: int, y: int) -> None:
    """Single left-click."""
    import pyautogui

    log.debug("click(%d, %d)", x, y)
    pyautogui.click(x, y)


def paste_text(text: str) -> None:
    """Copy *text* to the clipboard then paste with Ctrl+V — instant, no per-char delay."""
    import pyperclip
    import pyautogui

    log.debug("pasting %d chars via clipboard", len(text))
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")


def hotkey(*keys: str) -> None:
    """Press a key combination, e.g. hotkey('ctrl', 's')."""
    import pyautogui

    log.debug("hotkey(%s)", "+".join(keys))
    pyautogui.hotkey(*keys)


def press(key: str) -> None:
    """Press a single named key."""
    import pyautogui

    log.debug("press(%s)", key)
    pyautogui.press(key)


# ---------------------------------------------------------------------------
# Window management
# ---------------------------------------------------------------------------

def show_desktop() -> None:
    """
    Minimize ALL windows so the desktop is visible before grounding.

    Shell.MinimizeAll() (COM) is a true OS-level minimize that doesn't toggle and
    that VS Code can't intercept the way it does Win+D.  Any window that survives
    is then minimized directly.
    """
    import subprocess
    import pygetwindow as gw

    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(New-Object -ComObject Shell.Application).MinimizeAll()"],
        capture_output=True,
    )
    time.sleep(0.4)

    for w in gw.getAllWindows():
        try:
            if w.title and w.isActive:
                w.minimize()
        except Exception:
            pass
    time.sleep(0.2)


def window_is_open(title_substring: str) -> bool:
    """Return True if any open window's title contains *title_substring*."""
    try:
        import pygetwindow as gw  # local import

        return any(title_substring.lower() in w.title.lower() for w in gw.getAllWindows())
    except Exception:
        log.debug("pygetwindow query failed", exc_info=True)
        return False


def wait_for_window(
    title_substring: str,
    timeout: float,
    interval: float = 0.5,
) -> bool:
    """Poll until a window matching *title_substring* appears or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if window_is_open(title_substring):
            return True
        time.sleep(interval)
    return False


def wait_for_window_gone(
    title_substring: str,
    timeout: float,
    interval: float = 0.5,
) -> bool:
    """
    Poll until NO window matching *title_substring* remains, or *timeout* expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not window_is_open(title_substring):
            return True
        time.sleep(interval)
    return not window_is_open(title_substring)


# ---------------------------------------------------------------------------
# Desktop focus (Win32) — make the desktop the foreground window so a
# double-click on an icon launches instead of being eaten by activation.
# ---------------------------------------------------------------------------

SPI_GETWORKAREA = 0x0030


def _user32():
    """Return user32 with HWND-safe restypes set (avoids 32-bit handle truncation)."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowExW.restype = wintypes.HWND
    user32.FindWindowExW.argtypes = [wintypes.HWND, wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    return user32


def _find_desktop() -> tuple[int, int]:
    """
    Return (top_level_hwnd, listview_hwnd); either may be 0.
    """
    user32 = _user32()
    progman = user32.FindWindowW("Progman", None) or 0
    defview = user32.FindWindowExW(progman, None, "SHELLDLL_DefView", None) or 0
    top = progman

    if not defview:
        worker = 0
        while True:
            worker = user32.FindWindowExW(None, worker, "WorkerW", None) or 0
            if not worker:
                break
            d = user32.FindWindowExW(worker, None, "SHELLDLL_DefView", None) or 0
            if d:
                defview = d
                top = worker
                break

    listview = (user32.FindWindowExW(defview, None, "SysListView32", None) or 0) if defview else 0
    return int(top), int(listview)


def focus_desktop() -> bool:
    """
    Make the desktop the foreground window and verify it took.
    Returns True if the desktop is now foreground, False otherwise.
    """
    user32 = _user32()
    top, listview = _find_desktop()
    target = listview or top
    if not target:
        return False

    user32.SetForegroundWindow(target)
    time.sleep(0.1)
    fg = int(user32.GetForegroundWindow() or 0)
    return fg != 0 and fg in (top, listview)


def get_work_area() -> tuple[int, int, int, int]:
    """Return the primary work area (left, top, right, bottom) — screen minus taskbar."""
    import ctypes
    from ctypes import wintypes

    rect = wintypes.RECT()
    ok = ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
    if not ok:
        w = ctypes.windll.user32.GetSystemMetrics(0)
        h = ctypes.windll.user32.GetSystemMetrics(1)
        return 0, 0, int(w), int(h)
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def close_window(title_substring: str) -> bool:
    import pygetwindow as gw

    wins = [w for w in gw.getAllWindows() if title_substring.lower() in w.title.lower()]
    if not wins:
        return False

    closed_any = False
    for w in wins:
        try:
            w.close()
            time.sleep(0.3)
            closed_any = True
        except Exception:
            log.warning("close_window(%r) failed on one window", title_substring, exc_info=True)

    return closed_any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

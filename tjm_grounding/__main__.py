"""Entry point: python -m tjm_grounding"""
from __future__ import annotations

import atexit
import logging
import sys

from tjm_grounding import config


def _release_all_keys() -> None:
    try:
        import pyautogui
        for key in ("ctrl", "alt", "shift", "win"):
            pyautogui.keyUp(key)
        pyautogui.mouseUp(button="left")
        pyautogui.mouseUp(button="right")
    except Exception:
        pass


def main() -> None:
    atexit.register(_release_all_keys)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(levelname)s | %(message)s",
    )
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if not config.OPENROUTER_API_KEY:
        print(
            "ERROR: OPENROUTER_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  $env:OPENROUTER_API_KEY = 'sk-or-...'\n",
            file=sys.stderr,
        )
        sys.exit(1)

    from tjm_grounding.grounder import ApiVlmGrounder
    from tjm_grounding.workflow import run

    grounder = ApiVlmGrounder()
    run(grounder)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""Keep the public demo on Streamlit Community Cloud awake by visiting it.

Community Cloud puts an app to sleep after 12 hours without a visitor, and the
next visitor then has to press a wake-up button and wait for the restart (about
two minutes for this app). A plain HTTP request does not count as a visit: the
root URL is answered by the Community Cloud shell (HTTP 200 whether the app is
awake or asleep), while the app itself runs in an iframe at ``/~/+/`` and talks
to the server over a WebSocket. This script therefore opens the page in headless
Chromium (Playwright), presses the wake-up button when the sleep screen is
shown, waits until the app has rendered inside the iframe and stays there for a
few seconds. GitHub Actions runs it every few hours
(``.github/workflows/keep-demo-awake.yml``); it can also be run by hand::

    python -m pip install playwright && python -m playwright install chromium
    python demo/keepalive.py [--url URL] [--state FILE] [--screenshot FILE]

Exit status 0 means the app was confirmed awake (it was already awake, or it
was woken up here); 1 means it could not be confirmed within ``--wake-timeout``,
in which case a screenshot is written for diagnosis when ``--screenshot`` is
given. ``--state`` keeps the browser cookies between runs so that these visits
appear as a single viewer in the Community Cloud analytics (the proxy assigns
the viewer's pseudonym through a cookie).

Only the constants and ``health()`` are needed by the tests; Playwright is
imported inside ``visit()`` so that this module can be imported without it.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

DEMO_URL = "https://climcanvas.streamlit.app/"
# The Streamlit app itself: an iframe inside the Community Cloud shell.
APP_FRAME_PATH = "/~/+/"
# Answers "ok" while the app container is running (a direct request to the container).
HEALTH_PATH = APP_FRAME_PATH + "_stcore/health"
# The button on the sleep screen of the shell (outer frame), and its label as a fallback.
WAKE_BUTTON = '[data-testid="wakeup-button-viewer"]'
WAKE_BUTTON_LABEL = re.compile(r"get this app back up", re.IGNORECASE)
# Rendered inside the app frame once the app script has run (not just once the frontend loaded).
APP_READY = '[data-testid="stSidebar"]'
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/140.0.0.0 Safari/537.36")


def health(url: str, timeout: float = 20.0) -> str:
    """Body of the health endpoint ("ok" while the app runs) or a short error description."""
    req = urllib.request.Request(url.rstrip("/") + HEALTH_PATH, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", "replace").strip()
            if len(body) <= 40:
                return body or f"HTTP {resp.status}"
            # the shell answers with its HTML page when the request does not reach the container
            return f"HTTP {resp.status} ({len(body)}+ bytes, not the health text)"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        return f"error: {exc}"


def app_frame(page):
    """The frame that holds the Streamlit app, or None while the shell has not created it."""
    for frame in page.frames:
        if frame is not page.main_frame and APP_FRAME_PATH in frame.url:
            return frame
    return None


def wake_button(page):
    """The wake-up button of the sleep screen when it is shown, else None."""
    for locator in (page.locator(WAKE_BUTTON), page.get_by_role("button", name=WAKE_BUTTON_LABEL)):
        try:
            if locator.count() and locator.first.is_visible():
                return locator.first
        except Exception:  # the shell is re-rendering; look again on the next round
            pass
    return None


def visit(url: str, state: pathlib.Path | None, screenshot: pathlib.Path | None,
          wake_timeout: float, dwell: float) -> int:
    """Open the demo, wake it up if needed, stay ``dwell`` seconds. Returns the exit status."""
    from playwright.sync_api import sync_playwright

    started = time.monotonic()
    print(f"health before: {health(url)}", flush=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            user_agent=USER_AGENT, viewport={"width": 1280, "height": 900},
            storage_state=str(state) if state is not None and state.is_file() else None)
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        woke = False
        ready = False
        deadline = started + wake_timeout
        while time.monotonic() < deadline:
            if not woke:
                button = wake_button(page)
                if button is not None:
                    button.click()
                    woke = True
                    print(f"asleep: pressed the wake-up button after {time.monotonic() - started:.0f}s",
                          flush=True)
            frame = app_frame(page)
            if frame is not None:
                try:
                    if frame.locator(APP_READY).first.is_visible():
                        ready = True
                        break
                except Exception:  # the frame is being replaced while the app restarts
                    pass
            page.wait_for_timeout(2_000)
        elapsed = time.monotonic() - started
        if ready:
            print(f"{'woken up' if woke else 'awake'}: app rendered after {elapsed:.0f}s, "
                  f"staying {dwell:.0f}s", flush=True)
            page.wait_for_timeout(int(dwell * 1000))
        else:
            print(f"FAILED: app not rendered within {wake_timeout:.0f}s "
                  f"(wake-up button pressed: {woke})", flush=True)
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot), full_page=True)
                print(f"screenshot: {screenshot}", flush=True)
        if state is not None:
            state.parent.mkdir(parents=True, exist_ok=True)
            context.storage_state(path=str(state))
        browser.close()
    print(f"health after: {health(url)}", flush=True)
    return 0 if ready else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEMO_URL, help=f"app URL (default {DEMO_URL})")
    ap.add_argument("--state", type=pathlib.Path,
                    help="JSON file that keeps the browser cookies between runs")
    ap.add_argument("--screenshot", type=pathlib.Path,
                    help="PNG written when the app could not be confirmed awake")
    ap.add_argument("--wake-timeout", type=float, default=420.0,
                    help="seconds to wait for the app to render (default 420; waking up takes "
                         "about 2 minutes)")
    ap.add_argument("--dwell", type=float, default=20.0,
                    help="seconds to stay on the page once the app has rendered (default 20)")
    args = ap.parse_args(argv)
    return visit(args.url, args.state, args.screenshot, args.wake_timeout, args.dwell)


if __name__ == "__main__":
    sys.exit(main())

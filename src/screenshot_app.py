"""Take the two app screenshots used in the lesson and report (figures 46 and 47).

Optional tool, NOT part of the app and NOT in requirements.txt. It needs:  pip install playwright   (uses your installed Chrome).
Start the app first in another terminal:  streamlit run app/streamlit_app.py --server.port 8599
Then:  python src/screenshot_app.py [port]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FIGURES = Path(__file__).resolve().parent.parent / "reports" / "figures"
SHOTS = [("", "46_app_screenshot_match.png", 2400), ("season_table", "47_app_screenshot_season.png", 2700)]


def main(port="8599"):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        for path, name, height in SHOTS:
            page = browser.new_page(viewport={"width": 1280, "height": height})   # tall window: the whole page is visible without scrolling
            page.goto(f"http://localhost:{port}/{path}")
            page.wait_for_selector(".hero", timeout=60000)
            page.wait_for_timeout(5000)                                           # let the charts finish drawing
            page.screenshot(path=str(FIGURES / name))
            print("saved", FIGURES / name)
        browser.close()


if __name__ == "__main__":
    main(*sys.argv[1:])

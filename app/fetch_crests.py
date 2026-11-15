"""One-off helper: download the 20 club crests as small transparent PNGs into data/app/team_crests/.

Source: the English Wikipedia club articles' infobox image (the crest), fetched as a PNG thumbnail through the public
MediaWiki API (commons.wikimedia.org is not reachable from this machine, and most of these crests are not on Commons
at all but local "non-free" files of English Wikipedia). It makes 20 polite requests, then never runs again.

LICENSING NOTE: club crests are trademarked, non-free logos. data/app/team_crests/SOURCES.md records where each file came
from. The PNGs are git-ignored until the project owner decides whether to publish them (the app works without them).

Run once:  python app/fetch_crests.py
"""
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

from team_colors import SLUG, WIKI_ARTICLE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "app" / "team_crests"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "pl-student-project/1.0 (university course project; educational)"}
SIZE = 128   # stored at 128 x 128 so the 96 px matchup card stays sharp on high-density screens


def polite_get(url, **kwargs):
    """GET that waits and retries when Wikimedia answers 429 (too many requests)."""
    for attempt in range(6):
        response = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait = int(response.headers.get("Retry-After", 30)) + 5 * attempt
        print(f"  (rate limited, waiting {wait}s)")
        time.sleep(wait)
    raise RuntimeError("still rate limited; try again later")


def crest_url(article):
    """The crest named in the club article's infobox (`| logo = X.svg`), as a PNG thumbnail URL, plus that file name."""
    wikitext = polite_get(API, params={"action": "parse", "page": article, "prop": "wikitext", "section": 0,
                                         "format": "json", "redirects": 1}, ).json()["parse"]["wikitext"]["*"]
    match = re.search(r"\|\s*(?:logo|image)\s*=\s*(?:\[\[)?(?:File:|Image:)?([^|\]\r\n]+?\.(?:svg|png|jpg|jpeg))", wikitext, re.I)
    file_name = match.group(1).strip()
    params = {"action": "query", "titles": "File:" + file_name, "prop": "imageinfo", "iiprop": "url", "iiurlwidth": SIZE * 2, "format": "json"}
    page = next(iter(polite_get(API, params=params).json()["query"]["pages"].values()))
    return page["imageinfo"][0]["thumburl"], file_name


def to_square_png(raw):
    """Fit the image inside a SIZE x SIZE transparent square (crests are not square)."""
    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    image.thumbnail((SIZE, SIZE), Image.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2), image)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Crest sources", "", "Fetched from the English Wikipedia club infobox images (see app/fetch_crests.py). "
             "These are trademarked, non-free club logos; they are for this educational project only and are git-ignored.", ""]
    for team, article in WIKI_ARTICLE.items():
        url, file_name = crest_url(article)
        image = to_square_png(polite_get(url).content)
        image.save(OUT / f"{SLUG[team]}.png")
        lines.append(f"- {team}: Wikipedia file `{file_name}` (article: {article})")
        print(f"{team:15s} {file_name}")
        time.sleep(4.0)
    (OUT / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

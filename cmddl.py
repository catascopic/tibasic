#!/usr/bin/env python3
"""
Parse One_Byte_Tokens_-_TI-Basic_Developer.htm, find all command links
(skipping 2-byte entries), fetch each page, strip it to plain body text,
and save to commands/<commandname>.
"""

import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from bs4 import BeautifulSoup

SOURCE_FILE = "webref/Graph Format Tokens - TI-Basic Developer.htm"
PREFIX = 'EF'
OUTPUT_DIR = Path("commands")
OUTPUT_DIR.mkdir(exist_ok=True)

SKIP_PATTERNS = [
    "variable-tokens",   # 2-byte token index pages
    "javascript:",       # inline JS links
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TIBasicScraper/1.0)"
}


def is_skippable(href: str, cell_text: str) -> bool:
    if not href or not href.startswith("http"):
        return True
    for pat in SKIP_PATTERNS:
        if pat in href:
            return True
    if "2-byte" in cell_text.lower():
        return True
    return False


def url_to_filename(href: str) -> str:
    """Turn a URL into a safe filename, keeping the path slug."""
    # Strip any #fragment
    href = href.split("#")[0]
    slug = href.rstrip("/").rsplit("/", 1)[-1]
    # Replace characters that are unsafe in filenames
    slug = re.sub(r'[<>:"/\\|?*]', '_', slug)
    return slug or "index"


def strip_page(html: str) -> str:
    """
    Extract just the meaningful content from a wikidot page:
    - Find the main content div (#page-content or similar)
    - Remove all tags, keeping text
    - Collapse whitespace
    """
    soup = BeautifulSoup(html, "html.parser")

    # Wikidot main content lives in #page-content
    content = (
        soup.find(id="page-content")
        or soup.find(id="main-content")
        or soup.find("article")
        or soup.find("body")
    )

    if content is None:
        return soup.get_text(separator="\n", strip=True)

    # Remove the site-wide "We're glad you came by" banner, which is always
    # the first <table> directly inside #page-content.
    first_table = content.find("table")
    if first_table and "We're glad you came" in first_table.get_text():
        first_table.decompose()

    # Remove nav, scripts, styles, footers, sidebars
    for tag in content.find_all(["script", "style", "nav", "footer",
                                  "aside", "noscript"]):
        tag.decompose()

    # Remove wikidot toolbar/rating divs by class
    for tag in content.find_all(True):
        cls = " ".join(tag.get("class", []))
        if any(x in cls for x in ["page-rate", "page-tags", "page-watch",
                                    "printuser", "edit-", "odometer",
                                    "yui-navset", "code"]):
            # keep <div class="code"> — those are TI-Basic examples, very useful
            if "code" not in cls:
                tag.decompose()

    text = content.get_text(separator="\n", strip=True)
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch(url: str) -> str | None:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            charset = "utf-8"
            ct = resp.headers.get_content_charset()
            if ct:
                charset = ct
            return resp.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} — skipping")
    except urllib.error.URLError as e:
        print(f"  URL error: {e.reason} — skipping")
    except Exception as e:
        print(f"  Error: {e} — skipping")
    return None


def collect_links(soup: BeautifulSoup) -> list[tuple[str, str, str]]:
    """Return [(hex_byte, label, href), ...] for all command cells, deduped.

    The token reference uses two side-by-side tables:
      table 3: columns 0–7  (low nibbles 0x_0 – 0x_7)
      table 4: columns 8–F  (low nibbles 0x_8 – 0x_F)
    Each table has a header row (row 0: title span, row 1: col nibble labels)
    followed by 16 data rows whose first cell is the high nibble (0–F).
    """
    all_tables = soup.find_all("table")
    # The two 18-row tables are the token tables
    token_tables = [t for t in all_tables if len(t.find_all("tr")) == 18]

    seen_hrefs: set[str] = set()
    links: list[tuple[str, str, str]] = []

    for table in token_tables:
        rows = table.find_all("tr")
        # Row 0: merged title; row 1: column-nibble headers; rows 2–17: data
        col_header_row = rows[1]
        col_nibbles = [
            c.get_text(strip=True)
            for c in col_header_row.find_all(["td", "th"])
        ]
        # col_nibbles[0] is the empty corner cell; col_nibbles[1..] are '0','1',...

        for data_row in rows[2:]:
            cells = data_row.find_all(["td", "th"])
            if not cells:
                continue
            row_nibble = cells[0].get_text(strip=True)  # '0' – 'F'

            for col_idx, cell in enumerate(cells[1:], start=1):
                col_nibble = col_nibbles[col_idx] if col_idx < len(col_nibbles) else "?"
                byte_hex = f"{col_nibble}{row_nibble}".upper()

                cell_text = cell.get_text(strip=True)
                anchor = cell.find("a")
                if not anchor:
                    continue
                href = anchor.get("href", "")
                label = anchor.get_text(strip=True)

                if is_skippable(href, cell_text):
                    continue

                base = href.split("#")[0]
                if base in seen_hrefs:
                    continue
                seen_hrefs.add(base)
                links.append((byte_hex, label, href))

    return links


def main():
    print(f"Parsing {SOURCE_FILE} …")
    with open(SOURCE_FILE, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    links = collect_links(soup)
    print(f"Found {len(links)} command links\n")

    for i, (byte_hex, label, href) in enumerate(links, 1):
        filename = f"{PREFIX}{byte_hex}_{url_to_filename(href)}.txt"
        out_path = OUTPUT_DIR / filename

        print(f"[{i}/{len(links)}] {byte_hex}  {label!r:20s} → {href}")

        if out_path.exists():
            print(f"  Already exists, skipping.")
            continue

        html = fetch(href)
        if html is None:
            continue

        text = strip_page(html)
        out_path.write_text(text, encoding="utf-8")
        print(f"  Saved to {out_path}  ({len(text)} chars)")

        # Be polite
        time.sleep(0.4)

    print("\nDone.")


if __name__ == "__main__":
    main()
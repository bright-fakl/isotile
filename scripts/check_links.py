"""Check the external (http and https) links of built documentation and of Markdown files.

Run: ``python scripts/check_links.py PATH [PATH ...] [--ignore REGEX ...] [--ignore-file FILE]``

``PATH`` is a file or a directory (searched for ``*.html`` and ``*.md``). Internal links are not checked here; the
strict mkdocs build covers them. A link passes with any 2xx status after redirects, and with 429 (rate limited, the
link is not known to be broken). Other failures and connection errors are retried 3 times with a growing wait before
the link counts as broken. Links that match an ignore pattern (a regular expression searched in the URL), loopback
addresses and links with template characters are skipped. The exit status is 1 if a link is broken.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = re.compile(r"https?://[^\s)\"'<>\]\\`]+")
SKIP = re.compile(r"[{}$*]|//(localhost|127\.0\.0\.1|\[::1\])([:/]|$)")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; isotile-link-check; +https://github.com/bright-fakl/isotile)"}


def collect(paths: list[Path]) -> dict[str, list[str]]:
    """Map each URL to the files that contain it."""
    files: list[Path] = []
    for path in paths:
        files += sorted(p for p in path.rglob("*") if p.suffix in {".html", ".md"}) if path.is_dir() else [path]
    found: dict[str, list[str]] = {}
    for file in files:
        for url in URL.findall(file.read_text(encoding="utf-8", errors="replace")):
            url = url.rstrip(".,;:").replace("&amp;", "&")
            found.setdefault(url, [])
            if str(file) not in found[url]:
                found[url].append(str(file))
    return found


def check(url: str, retries: int = 3, timeout: float = 30.0) -> str | None:
    """None if the link is fine, otherwise a short reason."""
    reason = ""
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)  # noqa: S310
            with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310
                return None
        except urllib.error.HTTPError as error:
            if error.code == 429:
                return None
            reason = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = f"{type(error).__name__}: {getattr(error, 'reason', error)}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    return reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--ignore", action="append", default=[], help="regular expression of URLs to skip")
    ap.add_argument("--ignore-file", type=Path, help="one regular expression per line; # starts a comment")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    for path in [p for p in args.paths if not p.exists()]:
        print(f"skipped, does not exist: {path}")
    args.paths = [p for p in args.paths if p.exists()]
    patterns = [re.compile(p) for p in args.ignore]
    if args.ignore_file:
        for line in args.ignore_file.read_text(encoding="utf-8").splitlines():
            line = line.split(" #")[0].strip()
            if line and not line.startswith("#"):
                patterns.append(re.compile(line))
    found = collect(args.paths)
    urls = sorted(u for u in found if not SKIP.search(u) and not any(p.search(u) for p in patterns))
    with ThreadPoolExecutor(args.workers) as pool:
        results = dict(zip(urls, pool.map(check, urls), strict=True))
    broken = {u: r for u, r in results.items() if r}
    for url, reason in broken.items():
        where = ", ".join(found[url][:3]) + (f" and {len(found[url]) - 3} more" if len(found[url]) > 3 else "")
        print(f"BROKEN {reason}: {url}  ({where})")
    print(f"{len(urls)} links checked, {len(found) - len(urls)} skipped, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

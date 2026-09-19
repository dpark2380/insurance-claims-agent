"""Download the canonical PDS/KFS sources into data/raw_pdfs/<insurer>/.

Reads data/sources.json (from discover_sources.py), downloads each PDF via
stdlib urllib (same approach as discovery -- no new HTTP dependency), and
writes/updates data/manifest.json: doc_id -> {insurer, product, doc_type,
filename, url, sha256, num_bytes}.

Run:  python -m ingest.download
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import zlib
from pathlib import Path

from .http import fetch as http_fetch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCES_PATH = DATA / "sources.json"
RAW_DIR = DATA / "raw_pdfs"
MANIFEST_PATH = DATA / "manifest.json"


def safe_filename(insurer: str, product: str, doc_type: str, url: str) -> str:
    ext = ".pdf"
    base = f"{insurer}-{product}-{doc_type}"
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-").lower()
    return base + ext


def doc_id_for(insurer: str, product: str, doc_type: str) -> str:
    return f"{insurer}-{product}-{doc_type}".lower()


def maybe_decompress(content: bytes) -> bytes:
    # urllib doesn't auto-decompress; some servers gzip/deflate the response
    # regardless of whether we advertised Accept-Encoding. Detect by magic bytes.
    if content[:2] == b"\x1f\x8b":
        return gzip.decompress(content)
    if content[:2] in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
        return zlib.decompress(content)
    return content


def download_one(url: str) -> bytes | None:
    content, _headers = http_fetch(url, timeout=30)
    if content is None:
        return None
    return maybe_decompress(content)


def main():
    if not SOURCES_PATH.exists():
        raise SystemExit(f"{SOURCES_PATH} not found -- run discover_sources.py first")

    sources = json.loads(SOURCES_PATH.read_text())
    # Rebuilt fresh every run (never loaded from the previous manifest) --
    # every entry below is repopulated from the current sources.json regardless
    # of whether its file was skipped or re-downloaded, so an old manifest
    # entry for a source that's since been removed from sources.json can't
    # silently linger and keep feeding a stale PDF into the RAG corpus.
    manifest: dict[str, dict] = {}

    ok, failed = 0, 0
    for entry in sources:
        insurer, product, doc_type, url = (
            entry["insurer"], entry["product"], entry["doc_type"], entry["url"]
        )
        doc_id = doc_id_for(insurer, product, doc_type)
        insurer_dir = RAW_DIR / insurer
        insurer_dir.mkdir(parents=True, exist_ok=True)
        filename = safe_filename(insurer, product, doc_type, url)
        dest = insurer_dir / filename

        print(f"[{doc_id}] {url}")
        if dest.exists():
            # Already downloaded -- reuse the file on disk instead of re-fetching
            # over the network (these are static policy documents that don't
            # change mid-session; pass --force-redownload to override).
            content = dest.read_bytes()
            print(f"  skipped (already on disk, {len(content):,} bytes) -> {dest.relative_to(ROOT)}")
        else:
            time.sleep(1)
            content = download_one(url)
            if content is None:
                failed += 1
                continue
            if not content.startswith(b"%PDF"):
                print(f"  WARNING: response doesn't look like a PDF (first bytes: {content[:20]!r}) -- skipping")
                failed += 1
                continue
            dest.write_bytes(content)
            print(f"  saved {len(content):,} bytes -> {dest.relative_to(ROOT)}")

        sha256 = hashlib.sha256(content).hexdigest()
        manifest[doc_id] = {
            "insurer": insurer,
            "product": product,
            "doc_type": doc_type,
            "filename": filename,
            "path": str(dest.relative_to(ROOT)),
            "url": url,
            "sha256": sha256,
            "num_bytes": len(content),
        }
        ok += 1

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\ndownloaded {ok} ok, {failed} failed. manifest -> {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

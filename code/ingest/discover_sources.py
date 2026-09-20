"""Discover and dedupe AU home-insurance PDS/KFS PDF links from insurer sites.

Standard library only. Two steps:
  1. scrape each insurer's public policy-document page for PDF links
  2. dedupe to one canonical (insurer, product, doc_type) entry each,
     keeping the most recently dated PDS and any KFS, skipping TMD/SPDS/AIG
     for v1 (see rag-ai-project-outline.md, Phase 1)

Run:  python -m ingest.discover_sources
Writes: ../../data/sources.json
"""

from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

from .http import fetch as http_fetch

SOURCES = {
    "Allianz": "https://www.allianz.com.au/my-allianz/policy-information/policy-documents.html",
    "AAMI-Building": "https://www.aami.com.au/policy-documents/home-building-insurance",
    "AAMI-Contents": "https://www.aami.com.au/policy-documents/home-contents-insurance",
    "RealInsurance": "https://www.realinsurance.com.au/product-disclosure-statements",
    "RAA": "https://www.raa.com.au/help-centre/insurance-support/pds-and-fact-sheets/home-and-contents-insurance-product-disclosure-statement",
    "NRMA": "https://www.nrma.com.au/policy-booklets",
    # Added in the corpus expansion pass. Youi (JS-rendered page, no static PDF
    # links) and QBE (active 403 on this landing page) were tried and excluded
    # deliberately -- not worth a headless-browser workaround for one insurer,
    # and a 403 is a site's own decision to block automated access, respected
    # the same way afca.org.au's bot-detection was earlier in this project.
    "Suncorp": "https://www.suncorp.com.au/insurance/policy-documents.html",
    "CGU": "https://www.cgu.com.au/policy-booklets",
    "BudgetDirect": "https://www.autogeneral.com.au/customers/find-pds/budd/home/",
    "GIO": "https://www.gio.com.au/policy-documents/home-contents.html",
    "Woolworths": "https://insurance.everyday.com.au/home-insurance/useful-documents.html",
    "RACV": "https://www.racv.com.au/insurance/policy-documents/home.html",
    "RACQ": "https://www.racq.com.au/insurance/insurance-disclosure-documents",
    # Coles (home PDS/KFS hosted off this landing page, only FSG/TMD found
    # here), Australia Post (home insurance discontinued Sept 2025, no home
    # docs left on this page), and Bank of Melbourne (home PDS not linked
    # from this particular sub-page) were also tried and excluded -- none
    # cleanly scrapable via this generic pattern without hardcoding a
    # one-off direct PDF URL, which isn't worth doing for one insurer each.
    "ING": "https://www.ing.com.au/help-and-support/documents-and-forms/insurance.html",
}

HOME_KEYWORDS = re.compile(r"home|building|contents|dwelling|landlord|strata", re.I)
EXCLUDE_KEYWORDS = re.compile(r"\b(car|motor|auto|vehicle|travel|pet|boat|caravan|life|health)\b", re.I)
# "Summary of changes" docs are short change-logs, not the actual policy text -- exclude
# them so they never win a dedupe tie against the real PDS/KFS document.
NOT_SUBSTANTIVE = re.compile(r"summary of change", re.I)

DOC_TYPE_PATTERNS = [
    ("KFS", re.compile(r"kfs|key.?fact", re.I)),
    ("TMD", re.compile(r"tmd|target.?market", re.I)),
    ("SPDS", re.compile(r"spds|supplementary", re.I)),
    ("AIG", re.compile(r"aig|additional.?information|asic.?instrument|assist.?terms", re.I)),
    ("PDS", re.compile(r"pds|product.?disclosure", re.I)),
]

PRODUCT_PATTERNS = [
    ("Landlord", re.compile(r"landlord", re.I)),
    ("Building", re.compile(r"\bbuilding\b", re.I)),
    ("Contents", re.compile(r"\bcontents\b", re.I)),
    ("Home", re.compile(r"\bhome\b", re.I)),
]

# Require a plausible month (01-12) and day (01-31) when present, not just any
# two digits -- otherwise an unrelated 4+ digit product/version code starting
# with "20" (e.g. a doc reference like "20231045") gets accepted as a fake
# date with month=10, day=45, and can silently outrank a genuinely newer file.
DATE_PATTERN = re.compile(r"(20\d{2})(?:[-_]?(0[1-9]|1[0-2]))?(?:[-_]?(0[1-9]|[12]\d|3[01]))?")

V1_DOC_TYPES = {"PDS", "KFS"}


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._href = href
                self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def fetch(url: str) -> str | None:
    content, headers = http_fetch(url, timeout=15)
    if content is None:
        return None
    charset = headers.get_content_charset() or "utf-8"
    return content.decode(charset, errors="replace")


def find_pdf_links(html: str, base_url: str) -> list[tuple[str, str]]:
    parser = LinkExtractor()
    parser.feed(html)
    out = []
    for href, text in parser.links:
        if ".pdf" not in href.lower():
            continue
        full = href if href.startswith("http") else urljoin(base_url, href)
        out.append((full, text))
    return out


def classify(url: str, text: str) -> tuple[str | None, str | None]:
    haystack = f"{url} {text}"
    doc_type = None
    for name, pat in DOC_TYPE_PATTERNS:
        if pat.search(haystack):
            doc_type = name
            break
    product = None
    for name, pat in PRODUCT_PATTERNS:
        if pat.search(haystack):
            product = name
            break
    return doc_type, product


def extract_date_key(url: str, text: str) -> str:
    m = DATE_PATTERN.search(f"{url} {text}")
    return m.group(0) if m else ""


def scrape_all() -> list[dict]:
    found = []
    for insurer, url in SOURCES.items():
        print(f"=== {insurer} ({url}) ===")
        time.sleep(2)
        html = fetch(url)
        if html is None:
            continue
        pdfs = find_pdf_links(html, url)
        home_pdfs = [
            (u, t) for u, t in pdfs
            if HOME_KEYWORDS.search(u + " " + t)
            and not EXCLUDE_KEYWORDS.search(u + " " + t)
            and not NOT_SUBSTANTIVE.search(u + " " + t)
        ]
        print(f"  total pdf links: {len(pdfs)}  home-related: {len(home_pdfs)}")
        for u, t in home_pdfs:
            doc_type, product = classify(u, t)
            found.append({
                "insurer": insurer.split("-")[0],
                "product": product or "Home",
                "doc_type": doc_type,
                "url": u,
                "text": t,
                "date_key": extract_date_key(u, t),
            })
    return found


def dedupe(entries: list[dict]) -> list[dict]:
    v1 = [e for e in entries if e["doc_type"] in V1_DOC_TYPES]
    groups: dict[tuple, list[dict]] = {}
    for e in v1:
        key = (e["insurer"], e["product"], e["doc_type"])
        groups.setdefault(key, []).append(e)

    canonical = []
    for key, group in groups.items():
        group.sort(key=lambda e: e["date_key"], reverse=True)
        canonical.append(group[0])
    canonical.sort(key=lambda e: (e["insurer"], e["product"], e["doc_type"]))
    return canonical


def main():
    entries = scrape_all()
    print(f"\ntotal home-related PDF links found: {len(entries)}")
    canonical = dedupe(entries)
    print(f"canonical v1 documents after dedupe (PDS+KFS only): {len(canonical)}")
    for e in canonical:
        print(f"  [{e['insurer']:12} {e['product']:9} {e['doc_type']:4}] {e['text'][:70]!r}")

    out_path = Path(__file__).resolve().parents[2] / "data" / "sources.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(canonical, f, indent=2)
    print(f"\nwrote {len(canonical)} canonical sources to {out_path}")


if __name__ == "__main__":
    main()

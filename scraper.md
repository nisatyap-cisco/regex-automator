for 'swift codes'- i want to extract data from https://bank-code.net/country/UNITED-STATES-%28US%29/100, https://www.theswiftcodes.com/united-states/, tell me the best tech stack that canscrapt data from here- and write that- chcek if that scrapes data and stores in xlsx or not.


reference for you- 
For these two sites, the best PoC stack is a “static HTML crawler” stack (no browser automation) because both pages are server-rendered and paginated (you can fetch /page/2/ etc.).

Best tech stack (fast + reliable + writes XLSX)

Python 3.10+

httpx: fast HTTP client + timeouts

selectolax: very fast HTML parsing (faster than BeautifulSoup)

tenacity: retries with exponential backoff

pandas + openpyxl: easy .to_excel() writing to .xlsx

(optional) rich: nice logs

This is better than Selenium/Playwright for a PoC because there’s no JS rendering requirement visible in the pages we opened.

A single scraper that crawls BOTH sources and saves to XLSX
requirements.txt
httpx>=0.27.0
selectolax>=0.3.21
tenacity>=8.2.3
pandas>=2.2.0
openpyxl>=3.1.2
swift_scrape_both.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import httpx
import pandas as pd
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


SWIFT_RE = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")  # ISO 9362-ish: 8 or 11


@dataclass
class SwiftRecord:
    source: str
    bank: str
    city: str
    branch: str
    swift_code: str
    url: str


class FetchError(Exception):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, FetchError)),
)
def fetch(client: httpx.Client, url: str) -> str:
    r = client.get(url, follow_redirects=True)
    if r.status_code != 200:
        raise FetchError(f"HTTP {r.status_code}: {url}")
    return r.text


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def find_last_page_num(html: str, pattern: str) -> Optional[int]:
    """
    Attempts to find the last page number from pagination links.
    pattern should capture a page number group, e.g. r"/page/(\\d+)/"
    """
    tree = HTMLParser(html)
    nums = []
    for a in tree.css("a"):
        href = a.attributes.get("href") or ""
        m = re.search(pattern, href)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                pass
    return max(nums) if nums else None


def iter_pages_theswiftcodes(base: str, html_first: str) -> Iterable[str]:
    # theswiftcodes uses /united-states/page/2/ style pagination :contentReference[oaicite:2]{index=2}
    last = find_last_page_num(html_first, r"/page/(\d+)/")
    yield base
    if last:
        for p in range(2, last + 1):
            yield base.rstrip("/") + f"/page/{p}/"
    else:
        # fallback crawl until empty
        p = 2
        while True:
            yield base.rstrip("/") + f"/page/{p}/"
            p += 1


def iter_pages_bankcodenet(base: str, html_first: str) -> Iterable[str]:
    # bank-code.net pagination appears as "2", "3", ... and URLs like .../100 etc. :contentReference[oaicite:3]{index=3}
    # We can crawl by following page-number links we see, but simplest:
    # - parse numeric page links and hit them all
    tree = HTMLParser(html_first)
    page_urls = {base}

    for a in tree.css("a"):
        href = a.attributes.get("href") or ""
        if "UNITED-STATES-%28US%29" in href and "/country/" in href:
            # include additional pages like /2, /3, /100 etc
            if href.startswith("http"):
                page_urls.add(href)
            else:
                page_urls.add("https://bank-code.net" + href)

    # Stable ordering: base first, then others sorted
    yield base
    for u in sorted(page_urls):
        if u != base:
            yield u


def parse_theswiftcodes_page(html: str, url: str) -> list[SwiftRecord]:
    """
    The page includes a visible list with header:
    'ID Bank or Institution City Branch Swift Code' :contentReference[oaicite:4]{index=4}
    SWIFT codes appear in link text, e.g. PMFAUS66 / PMFAUS66HKG. :contentReference[oaicite:5]{index=5}
    """
    tree = HTMLParser(html)
    text = tree.text(separator="\n")
    lines = [clean(x) for x in text.splitlines() if clean(x)]

    recs: list[SwiftRecord] = []

    # Find the section after the header line; then parse rows starting with an integer ID.
    # Example row in extracted text:
    # "2 1ST PMF BANCORP LOS ANGELES, CA PMFAUS66HKG" :contentReference[oaicite:6]{index=6}
    for line in lines:
        # row begins with ID
        if not re.match(r"^\d+\s+", line):
            continue
        # Must contain a SWIFT code
        m = SWIFT_RE.search(line)
        if not m:
            continue
        swift = m.group(0)

        # Remove leading id and trailing swift
        line_wo_swift = line[:m.start()].strip()
        line_wo_swift = re.sub(r"^\d+\s+", "", line_wo_swift).strip()

        # Heuristic split:
        # last comma-state city chunk like "LOS ANGELES, CA"
        city_match = re.search(r"(.+?),\s*([A-Z]{2})\b", line_wo_swift)
        if not city_match:
            # Some rows might have city without ", ST"; keep whole as bank if uncertain
            bank = line_wo_swift
            city = ""
            branch = ""
        else:
            city_end = city_match.end()
            city = line_wo_swift[city_match.start():city_end].strip()
            bank = line_wo_swift[:city_match.start()].strip()
            # Anything after city could be branch (sometimes blank)
            branch = line_wo_swift[city_end:].strip()

        recs.append(
            SwiftRecord(
                source="theswiftcodes.com",
                bank=bank,
                city=city,
                branch=branch,
                swift_code=swift,
                url=url,
            )
        )

    # Dedup by swift_code
    dedup = {}
    for r in recs:
        dedup.setdefault(r.swift_code, r)
    return list(dedup.values())


def parse_bankcodenet_page(html: str, url: str) -> list[SwiftRecord]:
    """
    bank-code.net rows look like:
    '1 JPMORGAN CHASE BANK, N.A.-COLUMBUS CHASUS33XXX' in the rendered extraction :contentReference[oaicite:7]{index=7}
    and page /100 shows similar list rows. :contentReference[oaicite:8]{index=8}
    """
    tree = HTMLParser(html)
    text = tree.text(separator="\n")
    lines = [clean(x) for x in text.splitlines() if clean(x)]

    recs: list[SwiftRecord] = []
    for line in lines:
        if not re.match(r"^\d+\s+", line):
            continue
        m = SWIFT_RE.search(line)
        if not m:
            continue
        swift = m.group(0)

        # remove leading no. and trailing swift
        pre = line[:m.start()].strip()
        pre = re.sub(r"^\d+\s+", "", pre).strip()

        # In bank-code.net, bank-city often separated with '-' like "BANK NAME-CITY,ST" :contentReference[oaicite:9]{index=9}
        bank = pre
        city = ""
        branch = ""

        if "-" in pre:
            bank_part, city_part = pre.split("-", 1)
            bank = bank_part.strip()
            city = city_part.strip()

        recs.append(
            SwiftRecord(
                source="bank-code.net",
                bank=bank,
                city=city,
                branch=branch,
                swift_code=swift,
                url=url,
            )
        )

    dedup = {}
    for r in recs:
        dedup.setdefault((r.swift_code, r.bank, r.city), r)
    return list(dedup.values())


def scrape_all(
    theswift_base: str,
    bankcode_base: str,
    sleep_s: float = 1.0,
) -> list[SwiftRecord]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; swift-scraper/1.0)"
    }
    out: list[SwiftRecord] = []
    seen = set()

    with httpx.Client(headers=headers, timeout=httpx.Timeout(30.0)) as client:
        # theswiftcodes
        html1 = fetch(client, theswift_base)
        for page_url in iter_pages_theswiftcodes(theswift_base, html1):
            html = html1 if page_url.rstrip("/") == theswift_base.rstrip("/") else fetch(client, page_url)
            recs = parse_theswiftcodes_page(html, page_url)
            for r in recs:
                key = (r.source, r.swift_code)
                if key not in seen:
                    seen.add(key)
                    out.append(r)
            time.sleep(sleep_s)

        # bank-code.net
        html2 = fetch(client, bankcode_base)
        for page_url in iter_pages_bankcodenet(bankcode_base, html2):
            html = html2 if page_url == bankcode_base else fetch(client, page_url)
            recs = parse_bankcodenet_page(html, page_url)
            for r in recs:
                key = (r.source, r.swift_code, r.bank, r.city)
                if key not in seen:
                    seen.add(key)
                    out.append(r)
            time.sleep(sleep_s)

    return out


def main():
    theswift = "https://www.theswiftcodes.com/united-states/"
    bankcode = "https://bank-code.net/country/UNITED-STATES-%28US%29/100"

    rows = scrape_all(theswift, bankcode, sleep_s=1.0)

    df = pd.DataFrame([r.__dict__ for r in rows])
    # Basic normalization
    df["swift_code"] = df["swift_code"].str.upper().str.strip()

    out_xlsx = "swift_us_scrape.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.sort_values(["source", "swift_code"]).to_excel(xw, index=False, sheet_name="swift_codes")

    print(f"Wrote {len(df)} rows to {out_xlsx}")


if __name__ == "__main__":
    main()
This does store to XLSX

Yes — it writes an Excel workbook via pandas.ExcelWriter(..., engine="openpyxl") into swift_us_scrape.xlsx.

To verify locally after running:

python swift_scrape_both.py
ls -lh swift_us_scrape.xlsx
python -c "import pandas as pd; print(pd.read_excel('swift_us_scrape.xlsx').head())"
Notes about the two sources (practical + trust)

theswiftcodes.com explicitly documents SWIFT structure and provides a paginated listing for US.

bank-code.net also provides a paginated “List of all SWIFT / BIC codes…” and shows a disclaimer that the info should be verified for professional use.

TheSwiftCodes robots file (as indexed) broadly allows crawling except certain query params.


--
gemeric scraper
Recommended generic tech stack

Python 3.11

httpx + tenacity for fetch

selectolax for parse

pydantic for strong schemas (keeps it generic & safe)

pandas + openpyxl for XLSX output

typer for CLI

requirements.txt
httpx>=0.27.0
selectolax>=0.3.21
tenacity>=8.2.3
pydantic>=2.7.0
pandas>=2.2.0
openpyxl>=3.1.2
typer>=0.12.3
rich>=13.7.1
Generic implementation
Folder layout
generic_scraper/
├── main.py
├── requirements.txt
├── registry.py
├── core/
│   ├── fetch.py
│   ├── models.py
│   ├── writer.py
│   └── runner.py
└── adapters/
    ├── __init__.py
    ├── swift_theswiftcodes.py
    └── swift_bankcodenet.py
1) Core models (core/models.py)
from __future__ import annotations
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class Record(BaseModel):
    """Generic scraped record (schema-flexible via fields dict)."""
    identifier: str = Field(..., description="e.g., swift_bic")
    source: str = Field(..., description="adapter/source name")
    value: str = Field(..., description="the identifier value itself (e.g., BIC)")
    fields: Dict[str, Any] = Field(default_factory=dict)
    url: Optional[str] = None
2) Fetch with retries (core/fetch.py)
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class FetchError(Exception):
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, FetchError)),
)
def fetch_text(client: httpx.Client, url: str) -> str:
    r = client.get(url, follow_redirects=True)
    if r.status_code != 200:
        raise FetchError(f"HTTP {r.status_code} for {url}")
    return r.text
3) Writer to XLSX (core/writer.py)
from __future__ import annotations
from pathlib import Path
import pandas as pd
from core.models import Record

def write_xlsx(records: list[Record], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # flatten records to a tabular DataFrame
    rows = []
    for r in records:
        row = {
            "identifier": r.identifier,
            "source": r.source,
            "value": r.value,
            "url": r.url,
        }
        # merge dynamic fields
        for k, v in r.fields.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        # still write an empty file with headers
        df = pd.DataFrame(columns=["identifier", "source", "value", "url"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        # one combined sheet
        df.to_excel(xw, index=False, sheet_name="data")

        # optional: per-source sheets
        for src in sorted(df["source"].dropna().unique()):
            df[df["source"] == src].to_excel(xw, index=False, sheet_name=src[:31])
4) Adapter interface + runner (core/runner.py)
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Optional
import time
import httpx
from core.models import Record
from core.fetch import fetch_text

class Adapter(ABC):
    """Pluggable source adapter."""
    name: str

    @abstractmethod
    def supports(self, identifier: str, region: str) -> bool:
        ...

    @abstractmethod
    def seed_urls(self, identifier: str, region: str) -> list[str]:
        ...

    @abstractmethod
    def iter_page_urls(self, first_html: str, seed_url: str) -> Iterable[str]:
        ...

    @abstractmethod
    def parse(self, html: str, url: str, identifier: str, region: str) -> list[Record]:
        ...

def run_scrape(
    identifier: str,
    region: str,
    adapters: list[Adapter],
    sleep_s: float = 1.0,
) -> list[Record]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; generic-scraper/1.0)"}
    out: list[Record] = []
    seen = set()

    with httpx.Client(headers=headers, timeout=httpx.Timeout(30.0)) as client:
        for ad in adapters:
            if not ad.supports(identifier, region):
                continue

            for seed in ad.seed_urls(identifier, region):
                first_html = fetch_text(client, seed)
                for page_url in ad.iter_page_urls(first_html, seed):
                    html = first_html if page_url == seed else fetch_text(client, page_url)
                    recs = ad.parse(html, page_url, identifier, region)

                    for r in recs:
                        key = (r.source, r.value, tuple(sorted(r.fields.items())))
                        if key not in seen:
                            seen.add(key)
                            out.append(r)

                    time.sleep(sleep_s)

    return out
5) SWIFT adapters
adapters/swift_theswiftcodes.py
from __future__ import annotations
import re
from typing import Iterable, Optional
from selectolax.parser import HTMLParser
from core.models import Record
from core.runner import Adapter

BIC_RE = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

class TheSwiftCodesUS(Adapter):
    name = "theswiftcodes_us"

    def supports(self, identifier: str, region: str) -> bool:
        return identifier == "swift_bic" and region.lower() in {"usa", "us", "united_states", "united states"}

    def seed_urls(self, identifier: str, region: str) -> list[str]:
        return ["https://www.theswiftcodes.com/united-states/"]

    def iter_page_urls(self, first_html: str, seed_url: str) -> Iterable[str]:
        # infer /page/N/ links
        tree = HTMLParser(first_html)
        nums = []
        for a in tree.css("a"):
            href = a.attributes.get("href") or ""
            m = re.search(r"/page/(\d+)/", href)
            if m:
                nums.append(int(m.group(1)))
        last = max(nums) if nums else None

        yield seed_url
        if last:
            for p in range(2, last + 1):
                yield seed_url.rstrip("/") + f"/page/{p}/"
        else:
            p = 2
            while True:
                yield seed_url.rstrip("/") + f"/page/{p}/"
                p += 1

    def parse(self, html: str, url: str, identifier: str, region: str) -> list[Record]:
        tree = HTMLParser(html)
        lines = [clean(x) for x in tree.text(separator="\n").splitlines() if clean(x)]
        out: list[Record] = []

        for line in lines:
            if not re.match(r"^\d+\s+", line):
                continue
            m = BIC_RE.search(line)
            if not m:
                continue

            bic = m.group(0)
            before = re.sub(r"^\d+\s+", "", line[:m.start()].strip())

            # crude split bank/city/branch
            bank = before
            city = ""
            branch = ""
            m_city = re.search(r"(.+?,\s*[A-Z]{2})\b", before)
            if m_city:
                city = m_city.group(1).strip()
                bank = before[:m_city.start()].strip()
                branch = before[m_city.end():].strip()

            out.append(Record(
                identifier="swift_bic",
                source=self.name,
                value=bic,
                url=url,
                fields={"bank": bank, "city": city, "branch": branch, "country": "US"},
            ))

        return out
adapters/swift_bankcodenet.py
from __future__ import annotations
import re
from typing import Iterable
from selectolax.parser import HTMLParser
from core.models import Record
from core.runner import Adapter

BIC_RE = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

class BankCodeUS(Adapter):
    name = "bank_code_us"

    def supports(self, identifier: str, region: str) -> bool:
        return identifier == "swift_bic" and region.lower() in {"usa", "us", "united_states", "united states"}

    def seed_urls(self, identifier: str, region: str) -> list[str]:
        return ["https://bank-code.net/country/UNITED-STATES-%28US%29/100"]

    def iter_page_urls(self, first_html: str, seed_url: str) -> Iterable[str]:
        # collect all /country/UNITED-STATES... links found on first page
        tree = HTMLParser(first_html)
        urls = {seed_url}
        for a in tree.css("a"):
            href = a.attributes.get("href") or ""
            if "/country/UNITED-STATES-%28US%29" in href:
                if href.startswith("http"):
                    urls.add(href)
                else:
                    urls.add("https://bank-code.net" + href)
        yield seed_url
        for u in sorted(urls):
            if u != seed_url:
                yield u

    def parse(self, html: str, url: str, identifier: str, region: str) -> list[Record]:
        tree = HTMLParser(html)
        lines = [clean(x) for x in tree.text(separator="\n").splitlines() if clean(x)]
        out: list[Record] = []

        for line in lines:
            if not re.match(r"^\d+\s+", line):
                continue
            m = BIC_RE.search(line)
            if not m:
                continue

            bic = m.group(0)
            pre = re.sub(r"^\d+\s+", "", line[:m.start()].strip())

            bank, city = pre, ""
            if "-" in pre:
                bank, city = [x.strip() for x in pre.split("-", 1)]

            out.append(Record(
                identifier="swift_bic",
                source=self.name,
                value=bic,
                url=url,
                fields={"bank": bank, "city": city, "country": "US"},
            ))

        return out
6) Registry (registry.py)
from adapters.swift_theswiftcodes import TheSwiftCodesUS
from adapters.swift_bankcodenet import BankCodeUS

ALL_ADAPTERS = [
    TheSwiftCodesUS(),
    BankCodeUS(),
]
7) CLI orchestrator (main.py)
from __future__ import annotations
from pathlib import Path
import typer
from rich import print as rprint

from core.runner import run_scrape
from core.writer import write_xlsx
from registry import ALL_ADAPTERS

app = typer.Typer(add_completion=False)

@app.command()
def scrape(
    identifier: str = typer.Argument(..., help="e.g. swift_bic, pin_code"),
    region: str = typer.Argument(..., help="e.g. usa, india"),
    out: Path = typer.Option(Path("out.xlsx"), help="Output .xlsx path"),
    sleep_s: float = typer.Option(1.0, help="Delay between requests"),
):
    records = run_scrape(identifier=identifier, region=region, adapters=ALL_ADAPTERS, sleep_s=sleep_s)
    rprint(f"[green]Scraped[/green] {len(records)} records")
    write_xlsx(records, out)
    rprint(f"[bold]Wrote[/bold] {out}")

if __name__ == "__main__":
    app()

Run:

python main.py scrape swift_bic usa --out swift_usa.xlsx
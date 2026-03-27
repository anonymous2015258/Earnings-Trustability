"""Async EDGAR fetcher for earnings press releases via the Submissions API."""
import asyncio
import logging
import re
from datetime import date, timedelta
from typing import List, Optional

import httpx

from guidance_credibility.config import EDGAR_RATE_LIMIT_SECONDS, EDGAR_USER_AGENT
from guidance_credibility.models import TranscriptRecord

logger = logging.getLogger(__name__)

_EDGAR_BASE = "https://www.sec.gov"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# 8-K item 2.02 = "Results of Operations and Financial Condition" (earnings release)
_EARNINGS_ITEM = "2.02"


class EdgarFetcher:
    """Fetches earnings press releases from SEC EDGAR using the Submissions API."""

    def __init__(self) -> None:
        """Initialize the fetcher; CIK cache populated on first use."""
        self._cik_cache: dict = {}

    def get_earnings_transcripts(
        self, ticker: str, num_quarters: int = 8
    ) -> List[TranscriptRecord]:
        """Fetch up to num_quarters earnings press releases for ticker."""
        return asyncio.run(self._async_get_transcripts(ticker, num_quarters))

    async def _async_get_transcripts(
        self, ticker: str, num_quarters: int
    ) -> List[TranscriptRecord]:
        """Async implementation: CIK lookup → 8-K list → press release fetch."""
        async with httpx.AsyncClient(
            headers={"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"},
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            cik = await self._get_cik(client, ticker.upper())
            if not cik:
                logger.warning("Could not find CIK for ticker %s", ticker)
                return []

            earnings_filings = await self._get_earnings_filings(client, cik, num_quarters)
            results: List[TranscriptRecord] = []
            seen_quarters: set = set()

            for filing in earnings_filings:
                if len(results) >= num_quarters:
                    break
                filing_date: date = filing["filing_date"]
                accession: str = filing["accession"]
                quarter = self._date_to_quarter(filing_date)
                if quarter in seen_quarters:
                    continue
                seen_quarters.add(quarter)

                await asyncio.sleep(EDGAR_RATE_LIMIT_SECONDS)
                doc_url = await self._resolve_press_release_url(client, cik, accession)
                if not doc_url:
                    logger.warning("No press release found for %s %s", ticker, quarter)
                    continue

                await asyncio.sleep(EDGAR_RATE_LIMIT_SECONDS)
                text = await self._fetch_document(client, doc_url)
                if text is None:
                    logger.warning("Fetch failed for %s %s", ticker, quarter)
                    continue

                clean = self._clean_html(text)
                if len(clean.split()) < 200:
                    logger.warning(
                        "Skipping %s %s — too short (%d words)", ticker, quarter, len(clean.split())
                    )
                    continue

                record = TranscriptRecord(
                    ticker=ticker.upper(),
                    quarter=quarter,
                    filing_date=filing_date,
                    raw_text=clean,
                    accession_number=accession,
                    word_count=len(clean.split()),
                )
                results.append(record)
                logger.info("Fetched %s %s (%d words)", ticker, quarter, record.word_count)

        logger.info("Fetched %d earnings releases for %s", len(results), ticker)
        return results

    async def _get_cik(self, client: httpx.AsyncClient, ticker: str) -> Optional[str]:
        """Return zero-padded 10-digit CIK for a ticker, or None if not found."""
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        try:
            resp = await client.get(_TICKERS_URL)
            resp.raise_for_status()
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[ticker] = cik
                    logger.info("Resolved %s → CIK %s (%s)", ticker, cik, entry.get("title"))
                    return cik
        except Exception as exc:
            logger.warning("CIK lookup failed for %s: %s", ticker, exc)
        return None

    async def _get_earnings_filings(
        self, client: httpx.AsyncClient, cik: str, limit: int
    ) -> List[dict]:
        """Return list of {filing_date, accession} for earnings 8-Ks (item 2.02)."""
        url = _SUBMISSIONS_URL.format(cik=cik)
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Submissions API failed for CIK %s: %s", cik, exc)
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        items_list = recent.get("items", [])

        earnings = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            item_str = items_list[i] if i < len(items_list) else ""
            if _EARNINGS_ITEM not in item_str:
                continue
            try:
                filing_date = date.fromisoformat(dates[i])
            except (ValueError, IndexError):
                continue
            earnings.append({"filing_date": filing_date, "accession": accs[i]})
            if len(earnings) >= limit * 2:  # fetch extra, filter dupes by quarter later
                break

        # Sort chronologically descending (most recent first)
        earnings.sort(key=lambda x: x["filing_date"], reverse=True)
        return earnings

    async def _resolve_press_release_url(
        self, client: httpx.AsyncClient, cik: str, accession: str
    ) -> Optional[str]:
        """Find the EX-99.1 press release URL from the filing index."""
        acc_clean = accession.replace("-", "")
        cik_int = str(int(cik))  # strip leading zeros for URL
        index_url = (
            f"{_EDGAR_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/{accession}-index.htm"
        )
        try:
            resp = await client.get(index_url)
            if resp.status_code != 200:
                return None
            html = resp.text

            # Find EX-99.1 links first (press release)
            ex99_links = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.htm)"[^>]*>[^<]*(?:EX-99|99\.1|press release|earnings)',
                html,
                re.IGNORECASE,
            )
            if ex99_links:
                return f"{_EDGAR_BASE}{ex99_links[0]}"

            # Fallback: all non-index htm files, pick largest (likely press release)
            all_links = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.htm)"',
                html,
                re.IGNORECASE,
            )
            candidates = [l for l in all_links if "index" not in l.lower()]
            if not candidates:
                return None

            # Prefer the one with "pr" or "pressrelease" in filename
            for link in candidates:
                fname = link.split("/")[-1].lower()
                if any(kw in fname for kw in ["pr", "press", "release", "earnings", "result"]):
                    return f"{_EDGAR_BASE}{link}"

            # Final fallback: first non-index htm
            return f"{_EDGAR_BASE}{candidates[0]}"

        except Exception as exc:
            logger.warning("Index resolution failed for %s: %s", accession, exc)
            return None

    async def _fetch_document(
        self, client: httpx.AsyncClient, url: str
    ) -> Optional[str]:
        """Fetch a document URL and return its text, or None on error."""
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.warning("Document fetch failed %s: %s", url, exc)
            return None

    @staticmethod
    def _date_to_quarter(d: date) -> str:
        """Convert a date to quarter string like '2023Q3'."""
        quarter = (d.month - 1) // 3 + 1
        return f"{d.year}Q{quarter}"

    @staticmethod
    def _clean_html(text: str) -> str:
        """Strip HTML/XML tags, decode entities, normalize whitespace."""
        text = re.sub(
            r"<(script|style)[^>]*>.*?</(script|style)>", " ", text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", " ", text)
        entities = {
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&nbsp;": " ", "&#160;": " ", "&quot;": '"',
            "&#8220;": '"', "&#8221;": '"', "&#8212;": "—",
            "&#8226;": "•", "&#58;": ":",
        }
        for ent, char in entities.items():
            text = text.replace(ent, char)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

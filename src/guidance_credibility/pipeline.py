"""CredibilityPipeline — orchestrates all four agents end-to-end."""
import asyncio
import logging
from typing import Dict, List, Optional

from guidance_credibility import config
from guidance_credibility.agents.extractor import ExtractorAgent
from guidance_credibility.agents.scorer import ScorerAgent, _quarter_sort_key
from guidance_credibility.agents.sentiment import SentimentAgent
from guidance_credibility.agents.verifier import VerifierAgent
from guidance_credibility.db import DatabaseManager
from guidance_credibility.fetcher import EdgarFetcher
from guidance_credibility.models import CompanyScore, SignalRecord, Verdict

logger = logging.getLogger(__name__)


class CredibilityPipeline:
    """Four-agent pipeline: fetch → extract → verify → score → signal."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize all agents and open the database."""
        self.db = DatabaseManager(db_path or config.DB_PATH)
        self.db.initialize_schema()
        self.fetcher = EdgarFetcher()
        self.extractor = ExtractorAgent()
        self.verifier = VerifierAgent()
        self.scorer = ScorerAgent()
        self.sentiment = SentimentAgent()

    def run_extraction(self, ticker: str, quarter: str) -> int:
        """Extract claims for one ticker/quarter; returns count of new claims stored."""
        # Ensure transcript exists
        if not self.db.transcript_exists(ticker, quarter):
            transcripts = self.fetcher.get_earnings_transcripts(ticker, num_quarters=1)
            for t in transcripts:
                if t.quarter == quarter:
                    self.db.insert_transcript(t)
                    break
            else:
                logger.warning("No transcript found for %s %s", ticker, quarter)
                return 0

        # Skip if claims already extracted
        existing = self.db.get_claims_by_ticker_quarter(ticker, quarter)
        if existing:
            logger.info("Claims already extracted for %s %s — skipping", ticker, quarter)
            return 0

        transcript = self.db.get_transcript(ticker, quarter)
        if transcript is None:
            logger.warning("Transcript missing from DB for %s %s", ticker, quarter)
            return 0

        claims = self.extractor.extract_claims(transcript)
        for claim in claims:
            self.db.insert_claim(claim)
        logger.info("Stored %d claims for %s %s", len(claims), ticker, quarter)
        return len(claims)

    def run_verification(self, ticker: str, current_quarter: str) -> CompanyScore:
        """Verify pending prior claims against current_quarter transcript."""
        # Ensure current transcript
        if not self.db.transcript_exists(ticker, current_quarter):
            transcripts = self.fetcher.get_earnings_transcripts(ticker, num_quarters=1)
            for t in transcripts:
                if t.quarter == current_quarter:
                    self.db.insert_transcript(t)
                    break

        current_transcript = self.db.get_transcript(ticker, current_quarter)
        pending = self.db.get_pending_claims(ticker)

        # Filter to claims from quarters before current
        current_key = _quarter_sort_key(current_quarter)
        prior_pending = [
            c for c in pending if _quarter_sort_key(c.quarter) < current_key
        ]

        if prior_pending and current_transcript is not None:
            updated = self.verifier.verify_claims(prior_pending, current_transcript)
            for claim in updated:
                if claim.verdict != Verdict.PENDING:
                    self.db.update_claim_verdict(
                        claim.id,
                        claim.verdict.value,
                        claim.verdict_quarter or current_quarter,
                        claim.verdict_evidence or "",
                        claim.credibility_delta or 0.0,
                    )
        else:
            if not prior_pending:
                logger.info("No pending claims to verify for %s as of %s", ticker, current_quarter)
            if current_transcript is None:
                logger.warning("No current transcript for %s %s", ticker, current_quarter)

        return self.scorer.compute_score(ticker, current_quarter, self.db)

    def run_sentiment(self, ticker: str, quarter: str) -> SignalRecord:
        """Compute credibility-weighted sentiment signal for ticker/quarter."""
        transcript = self.db.get_transcript(ticker, quarter)
        if transcript is None:
            transcripts = self.fetcher.get_earnings_transcripts(ticker, num_quarters=1)
            for t in transcripts:
                if t.quarter == quarter:
                    self.db.insert_transcript(t)
                    transcript = t
                    break
        if transcript is None:
            raise ValueError(f"No transcript available for {ticker} {quarter}")

        score = self.db.get_score(ticker, quarter)
        topic_cred = score.topic_credibilities if score else {t.value: 0.5 for t in __import__("guidance_credibility.models", fromlist=["ClaimTopic"]).ClaimTopic}

        signal = self.sentiment.score_transcript(transcript, topic_cred)
        self.db.insert_or_replace_signal(signal)
        return signal

    def run_full_backtest(self, ticker: str, num_quarters: int = 8) -> Dict[str, list]:
        """Run the full extraction→verification→sentiment pipeline for all quarters."""
        logger.info("Starting full backtest for %s (%d quarters)", ticker, num_quarters)

        # Fetch all transcripts
        transcripts = self.fetcher.get_earnings_transcripts(ticker, num_quarters=num_quarters)
        for t in transcripts:
            self.db.insert_transcript(t)

        if not transcripts:
            logger.warning("No transcripts fetched for %s", ticker)
            return {"scores": [], "signals": []}

        quarters = sorted(
            [t.quarter for t in transcripts], key=_quarter_sort_key
        )
        logger.info("Processing %d quarters for %s: %s", len(quarters), ticker, quarters)

        # Extraction pass — ALL quarters first
        for quarter in quarters:
            try:
                self.run_extraction(ticker, quarter)
            except Exception as exc:
                logger.warning("Extraction failed for %s %s: %s", ticker, quarter, exc)

        # Verification + sentiment pass
        scores: List[CompanyScore] = []
        signals: List[SignalRecord] = []

        for quarter in quarters[1:]:  # skip first — nothing to verify against
            try:
                score = self.run_verification(ticker, quarter)
                scores.append(score)
            except Exception as exc:
                logger.warning("Verification failed for %s %s: %s", ticker, quarter, exc)

            try:
                signal = self.run_sentiment(ticker, quarter)
                signals.append(signal)
            except Exception as exc:
                logger.warning("Sentiment failed for %s %s: %s", ticker, quarter, exc)

        return {"scores": scores, "signals": signals}

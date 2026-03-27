"""SQLite database manager — no ORM, parameterized queries only."""
import sqlite3
import json
import logging
from datetime import date
from typing import List, Optional

from guidance_credibility.models import (
    Claim,
    ClaimTopic,
    CompanyScore,
    SignalRecord,
    SpecificityTier,
    TranscriptRecord,
    Verdict,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages all SQLite persistence for the credibility pipeline."""

    def __init__(self, db_path: str) -> None:
        """Initialize with path to the SQLite database file."""
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "DatabaseManager":
        """Open database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close database connection."""
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
            self.conn = None

    def _get_conn(self) -> sqlite3.Connection:
        """Return the active connection, opening one if needed."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def initialize_schema(self) -> None:
        """Create all four tables if they do not already exist."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                quarter TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                topic TEXT NOT NULL,
                specificity TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                source_sentence TEXT NOT NULL,
                verdict TEXT NOT NULL DEFAULT 'pending',
                verdict_quarter TEXT,
                verdict_evidence TEXT,
                credibility_delta REAL,
                embedding TEXT
            );

            CREATE TABLE IF NOT EXISTS company_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                quarter TEXT NOT NULL,
                score REAL NOT NULL,
                claims_evaluated INTEGER NOT NULL,
                fulfilled_count INTEGER NOT NULL,
                partial_count INTEGER NOT NULL,
                dropped_count INTEGER NOT NULL,
                walked_back_count INTEGER NOT NULL,
                dominant_topic_failure TEXT,
                topic_credibilities TEXT,
                UNIQUE(ticker, quarter)
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                ticker TEXT NOT NULL,
                quarter TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                accession_number TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                PRIMARY KEY(ticker, quarter)
            );

            CREATE TABLE IF NOT EXISTS signals (
                ticker TEXT NOT NULL,
                quarter TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                mgmt_revenue REAL NOT NULL,
                mgmt_margin REAL NOT NULL,
                mgmt_growth REAL NOT NULL,
                mgmt_macro REAL NOT NULL,
                qa_revenue REAL NOT NULL,
                qa_margin REAL NOT NULL,
                qa_growth REAL NOT NULL,
                qa_macro REAL NOT NULL,
                credibility_revenue REAL NOT NULL,
                credibility_margin REAL NOT NULL,
                credibility_growth REAL NOT NULL,
                credibility_macro REAL NOT NULL,
                cws_revenue REAL NOT NULL,
                cws_margin REAL NOT NULL,
                cws_growth REAL NOT NULL,
                cws_macro REAL NOT NULL,
                composite_cws REAL NOT NULL,
                sentiment_divergence REAL NOT NULL,
                raw_sentiment REAL NOT NULL,
                PRIMARY KEY(ticker, quarter)
            );
            """
        )
        conn.commit()
        logger.info("Database schema initialized at %s", self.db_path)

    # ------------------------------------------------------------------ claims

    def insert_claim(self, claim: Claim) -> None:
        """Insert a single claim; skip silently if id already exists."""
        conn = self._get_conn()
        embedding_json = json.dumps(claim.embedding) if claim.embedding is not None else None
        conn.execute(
            """
            INSERT OR IGNORE INTO claims
            (id, ticker, quarter, claim_text, topic, specificity, filing_date,
             source_sentence, verdict, verdict_quarter, verdict_evidence,
             credibility_delta, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.id,
                claim.ticker,
                claim.quarter,
                claim.claim_text,
                claim.topic.value,
                claim.specificity.value,
                claim.filing_date.isoformat(),
                claim.source_sentence,
                claim.verdict.value,
                claim.verdict_quarter,
                claim.verdict_evidence,
                claim.credibility_delta,
                embedding_json,
            ),
        )
        conn.commit()

    def _row_to_claim(self, row: sqlite3.Row) -> Claim:
        """Convert a database row to a Claim model instance."""
        embedding = json.loads(row["embedding"]) if row["embedding"] else None
        return Claim(
            id=row["id"],
            ticker=row["ticker"],
            quarter=row["quarter"],
            claim_text=row["claim_text"],
            topic=ClaimTopic(row["topic"]),
            specificity=SpecificityTier(row["specificity"]),
            filing_date=date.fromisoformat(row["filing_date"]),
            source_sentence=row["source_sentence"],
            verdict=Verdict(row["verdict"]),
            verdict_quarter=row["verdict_quarter"],
            verdict_evidence=row["verdict_evidence"],
            credibility_delta=row["credibility_delta"],
            embedding=embedding,
        )

    def get_claims_by_ticker_quarter(self, ticker: str, quarter: str) -> List[Claim]:
        """Return all claims for a ticker/quarter combination."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM claims WHERE ticker = ? AND quarter = ?",
            (ticker, quarter),
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def get_pending_claims(self, ticker: str) -> List[Claim]:
        """Return all pending (unverified) claims for a ticker."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM claims WHERE ticker = ? AND verdict = 'pending'",
            (ticker,),
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def update_claim_verdict(
        self,
        claim_id: str,
        verdict: str,
        verdict_quarter: str,
        evidence: str,
        delta: float,
    ) -> None:
        """Update the verdict fields for a single claim by id."""
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE claims
            SET verdict = ?, verdict_quarter = ?, verdict_evidence = ?,
                credibility_delta = ?
            WHERE id = ?
            """,
            (verdict, verdict_quarter, evidence, delta, claim_id),
        )
        conn.commit()

    # ---------------------------------------------------------- company_scores

    def insert_or_replace_score(self, score: CompanyScore) -> None:
        """Upsert a company credibility score for a ticker/quarter."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO company_scores
            (ticker, quarter, score, claims_evaluated, fulfilled_count,
             partial_count, dropped_count, walked_back_count,
             dominant_topic_failure, topic_credibilities)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                score.ticker,
                score.quarter,
                score.score,
                score.claims_evaluated,
                score.fulfilled_count,
                score.partial_count,
                score.dropped_count,
                score.walked_back_count,
                score.dominant_topic_failure,
                json.dumps(score.topic_credibilities),
            ),
        )
        conn.commit()

    def _row_to_score(self, row: sqlite3.Row) -> CompanyScore:
        """Convert a database row to a CompanyScore model instance."""
        topic_cred = json.loads(row["topic_credibilities"]) if row["topic_credibilities"] else {}
        return CompanyScore(
            ticker=row["ticker"],
            quarter=row["quarter"],
            score=row["score"],
            claims_evaluated=row["claims_evaluated"],
            fulfilled_count=row["fulfilled_count"],
            partial_count=row["partial_count"],
            dropped_count=row["dropped_count"],
            walked_back_count=row["walked_back_count"],
            dominant_topic_failure=row["dominant_topic_failure"],
            topic_credibilities=topic_cred,
        )

    def get_score_history(self, ticker: str) -> List[CompanyScore]:
        """Return all credibility scores for a ticker, ordered by quarter."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM company_scores WHERE ticker = ? ORDER BY quarter ASC",
            (ticker,),
        ).fetchall()
        return [self._row_to_score(r) for r in rows]

    def get_score(self, ticker: str, quarter: str) -> Optional[CompanyScore]:
        """Return the credibility score for a ticker/quarter, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM company_scores WHERE ticker = ? AND quarter = ?",
            (ticker, quarter),
        ).fetchone()
        return self._row_to_score(row) if row else None

    # ------------------------------------------------------------ transcripts

    def insert_transcript(self, record: TranscriptRecord) -> None:
        """Insert a transcript, skipping if the ticker/quarter already exists."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR IGNORE INTO transcripts
            (ticker, quarter, filing_date, raw_text, accession_number, word_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.ticker,
                record.quarter,
                record.filing_date.isoformat(),
                record.raw_text,
                record.accession_number,
                record.word_count,
            ),
        )
        conn.commit()

    def get_transcript(self, ticker: str, quarter: str) -> Optional[TranscriptRecord]:
        """Return the stored transcript for a ticker/quarter, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM transcripts WHERE ticker = ? AND quarter = ?",
            (ticker, quarter),
        ).fetchone()
        if row is None:
            return None
        return TranscriptRecord(
            ticker=row["ticker"],
            quarter=row["quarter"],
            filing_date=date.fromisoformat(row["filing_date"]),
            raw_text=row["raw_text"],
            accession_number=row["accession_number"],
            word_count=row["word_count"],
        )

    def transcript_exists(self, ticker: str, quarter: str) -> bool:
        """Return True if a transcript for ticker/quarter is already stored."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM transcripts WHERE ticker = ? AND quarter = ?",
            (ticker, quarter),
        ).fetchone()
        return row is not None

    # --------------------------------------------------------------- signals

    def insert_or_replace_signal(self, signal: SignalRecord) -> None:
        """Upsert a signal record for a ticker/quarter."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO signals
            (ticker, quarter, filing_date,
             mgmt_revenue, mgmt_margin, mgmt_growth, mgmt_macro,
             qa_revenue, qa_margin, qa_growth, qa_macro,
             credibility_revenue, credibility_margin,
             credibility_growth, credibility_macro,
             cws_revenue, cws_margin, cws_growth, cws_macro,
             composite_cws, sentiment_divergence, raw_sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.ticker,
                signal.quarter,
                signal.filing_date.isoformat(),
                signal.mgmt_revenue,
                signal.mgmt_margin,
                signal.mgmt_growth,
                signal.mgmt_macro,
                signal.qa_revenue,
                signal.qa_margin,
                signal.qa_growth,
                signal.qa_macro,
                signal.credibility_revenue,
                signal.credibility_margin,
                signal.credibility_growth,
                signal.credibility_macro,
                signal.cws_revenue,
                signal.cws_margin,
                signal.cws_growth,
                signal.cws_macro,
                signal.composite_cws,
                signal.sentiment_divergence,
                signal.raw_sentiment,
            ),
        )
        conn.commit()

    def _row_to_signal(self, row: sqlite3.Row) -> SignalRecord:
        """Convert a database row to a SignalRecord model instance."""
        return SignalRecord(
            ticker=row["ticker"],
            quarter=row["quarter"],
            filing_date=date.fromisoformat(row["filing_date"]),
            mgmt_revenue=row["mgmt_revenue"],
            mgmt_margin=row["mgmt_margin"],
            mgmt_growth=row["mgmt_growth"],
            mgmt_macro=row["mgmt_macro"],
            qa_revenue=row["qa_revenue"],
            qa_margin=row["qa_margin"],
            qa_growth=row["qa_growth"],
            qa_macro=row["qa_macro"],
            credibility_revenue=row["credibility_revenue"],
            credibility_margin=row["credibility_margin"],
            credibility_growth=row["credibility_growth"],
            credibility_macro=row["credibility_macro"],
            cws_revenue=row["cws_revenue"],
            cws_margin=row["cws_margin"],
            cws_growth=row["cws_growth"],
            cws_macro=row["cws_macro"],
            composite_cws=row["composite_cws"],
            sentiment_divergence=row["sentiment_divergence"],
            raw_sentiment=row["raw_sentiment"],
        )

    def get_signal(self, ticker: str, quarter: str) -> Optional[SignalRecord]:
        """Return the signal record for a ticker/quarter, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM signals WHERE ticker = ? AND quarter = ?",
            (ticker, quarter),
        ).fetchone()
        return self._row_to_signal(row) if row else None

    def get_signal_history(self, ticker: str) -> List[SignalRecord]:
        """Return all signal records for a ticker, ordered by quarter."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM signals WHERE ticker = ? ORDER BY quarter ASC",
            (ticker,),
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def get_all_signals_for_quarter(self, quarter: str) -> List[SignalRecord]:
        """Return all signal records for a given quarter across all tickers."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM signals WHERE quarter = ?",
            (quarter,),
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

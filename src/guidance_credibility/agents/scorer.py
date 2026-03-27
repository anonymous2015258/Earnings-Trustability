"""ScorerAgent — computes rolling credibility scores per company."""
import logging
from typing import List, Optional, Tuple

from guidance_credibility.config import (
    ROLLING_WINDOW_QUARTERS,
    SCORER_AMPLIFIER,
    SPECIFICITY_WEIGHTS,
    VERDICT_WEIGHTS,
)
from guidance_credibility.models import Claim, ClaimTopic, CompanyScore, Verdict

logger = logging.getLogger(__name__)


def _quarter_sort_key(quarter: str) -> int:
    """Convert '2023Q3' to a sortable integer."""
    try:
        return int(quarter[:4]) * 10 + int(quarter[5])
    except (IndexError, ValueError):
        return 0


class ScorerAgent:
    """Computes credibility scores from verified claims using a rolling window."""

    def compute_score(
        self, ticker: str, quarter: str, db  # DatabaseManager
    ) -> CompanyScore:
        """Compute and persist the credibility score for ticker as of quarter."""
        from guidance_credibility.db import DatabaseManager  # avoid circular at module level

        # Get all verified claims for this ticker across all quarters
        conn = db._get_conn()
        rows = conn.execute(
            "SELECT * FROM claims WHERE ticker = ? AND verdict != 'pending'",
            (ticker,),
        ).fetchall()
        from guidance_credibility.models import ClaimTopic as CT, SpecificityTier, Verdict as V
        import json as _json
        from datetime import date as _date

        verified_claims: List[Claim] = []
        for row in rows:
            emb = _json.loads(row["embedding"]) if row["embedding"] else None
            c = Claim(
                id=row["id"],
                ticker=row["ticker"],
                quarter=row["quarter"],
                claim_text=row["claim_text"],
                topic=CT(row["topic"]),
                specificity=SpecificityTier(row["specificity"]),
                filing_date=_date.fromisoformat(row["filing_date"]),
                source_sentence=row["source_sentence"],
                verdict=V(row["verdict"]),
                verdict_quarter=row["verdict_quarter"],
                verdict_evidence=row["verdict_evidence"],
                credibility_delta=row["credibility_delta"],
                embedding=emb,
            )
            verified_claims.append(c)

        # Rolling window: keep only claims from the last N distinct quarters
        current_key = _quarter_sort_key(quarter)
        all_quarters = sorted(
            set(c.quarter for c in verified_claims), key=_quarter_sort_key
        )
        if len(all_quarters) > ROLLING_WINDOW_QUARTERS:
            cutoff_quarters = set(all_quarters[-ROLLING_WINDOW_QUARTERS:])
        else:
            cutoff_quarters = set(all_quarters)

        window_claims = [c for c in verified_claims if c.quarter in cutoff_quarters]

        if not window_claims:
            score = CompanyScore(
                ticker=ticker,
                quarter=quarter,
                score=0.5,
                claims_evaluated=0,
                fulfilled_count=0,
                partial_count=0,
                dropped_count=0,
                walked_back_count=0,
                dominant_topic_failure=None,
                topic_credibilities={t.value: 0.5 for t in ClaimTopic},
            )
            db.insert_or_replace_score(score)
            return score

        # Credibility score — specificity-amplified weighted mean.
        #
        # credibility_delta (stored by VerifierAgent) already contains one factor
        # of spec_weight: delta = verdict_weight × spec_weight.
        # Multiplying by SCORER_AMPLIFIER[spec] applies a second factor, giving:
        #
        #   specific  → verdict × 1.0 × 1.0 = 1× numerator influence
        #   committed → verdict × 2.0 × 2.0 = 4× numerator influence
        #
        # raw_score range: ≈ [−1, +2] before normalization.
        # Normalized to [0, 1] via (raw + 1) / 2; clamped at both ends.
        numerator = sum(
            (c.credibility_delta or 0.0)
            * SCORER_AMPLIFIER.get(c.specificity.value, 1.0)
            for c in window_claims
        )
        denominator = sum(
            SCORER_AMPLIFIER.get(c.specificity.value, 1.0) for c in window_claims
        )
        raw_score = numerator / denominator if denominator > 0 else 0.0
        score_val = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))

        # Verdict counts
        fulfilled_count = sum(1 for c in window_claims if c.verdict == Verdict.FULFILLED)
        partial_count = sum(
            1 for c in window_claims if c.verdict == Verdict.PARTIALLY_FULFILLED
        )
        dropped_count = sum(
            1 for c in window_claims if c.verdict == Verdict.SILENTLY_DROPPED
        )
        walked_back_count = sum(
            1 for c in window_claims if c.verdict == Verdict.WALKED_BACK
        )

        # Topic credibilities
        topic_credibilities: dict = {}
        for topic in ClaimTopic:
            topic_claims = [c for c in window_claims if c.topic == topic]
            if not topic_claims:
                topic_credibilities[topic.value] = 0.5
            else:
                deltas = [c.credibility_delta or 0.0 for c in topic_claims]
                topic_credibilities[topic.value] = sum(deltas) / len(deltas)

        # Dominant failure topic
        failure_counts: dict = {}
        for c in window_claims:
            if c.verdict in (Verdict.WALKED_BACK, Verdict.SILENTLY_DROPPED):
                failure_counts[c.topic.value] = failure_counts.get(c.topic.value, 0) + 1
        dominant_topic_failure: Optional[str] = None
        if failure_counts:
            dominant_topic_failure = max(failure_counts, key=failure_counts.get)

        score = CompanyScore(
            ticker=ticker,
            quarter=quarter,
            score=score_val,
            claims_evaluated=len(window_claims),
            fulfilled_count=fulfilled_count,
            partial_count=partial_count,
            dropped_count=dropped_count,
            walked_back_count=walked_back_count,
            dominant_topic_failure=dominant_topic_failure,
            topic_credibilities=topic_credibilities,
        )
        db.insert_or_replace_score(score)
        logger.info(
            "Score for %s %s: %.3f (n=%d)", ticker, quarter, score_val, len(window_claims)
        )
        return score

    def get_score_trend(self, ticker: str, db) -> List[CompanyScore]:
        """Return score history for ticker with implicit QoQ delta in order."""
        history = db.get_score_history(ticker)
        return sorted(history, key=lambda s: _quarter_sort_key(s.quarter))

    def get_credibility_ranking(
        self, tickers: List[str], quarter: str, db
    ) -> List[Tuple[str, float]]:
        """Return (ticker, score) pairs sorted descending by score for a quarter."""
        results: List[Tuple[str, float]] = []
        for ticker in tickers:
            score_obj = db.get_score(ticker, quarter)
            score_val = score_obj.score if score_obj else 0.5
            results.append((ticker, score_val))
        return sorted(results, key=lambda x: x[1], reverse=True)

"""Pydantic v2 data models for the guidance credibility pipeline."""
from enum import Enum
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, List
import uuid


class SpecificityTier(str, Enum):
    """Tier of specificity for a forward-looking claim."""

    SPECIFIC = "specific"
    COMMITTED = "committed"


class ClaimTopic(str, Enum):
    """Topic category for a forward-looking claim."""

    REVENUE_GUIDANCE = "revenue_guidance"
    MARGIN_GUIDANCE = "margin_guidance"
    PRODUCT_LAUNCH = "product_launch"
    HEADCOUNT = "headcount"
    CAPEX = "capex"
    MARKET_SHARE = "market_share"
    MACRO_VIEW = "macro_view"
    OTHER = "other"


class Verdict(str, Enum):
    """Verdict on whether a prior claim materialized."""

    FULFILLED = "fulfilled"
    PARTIALLY_FULFILLED = "partially_fulfilled"
    SILENTLY_DROPPED = "silently_dropped"
    WALKED_BACK = "walked_back"
    PENDING = "pending"


class Claim(BaseModel):
    """A single forward-looking claim extracted from an earnings transcript."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    quarter: str  # "2023Q3"
    claim_text: str
    topic: ClaimTopic
    specificity: SpecificityTier
    filing_date: date
    source_sentence: str
    verdict: Verdict = Verdict.PENDING
    verdict_quarter: Optional[str] = None
    verdict_evidence: Optional[str] = None
    credibility_delta: Optional[float] = None
    embedding: Optional[List[float]] = None


class CompanyScore(BaseModel):
    """Rolling credibility score for a company in a given quarter."""

    ticker: str
    quarter: str
    score: float
    claims_evaluated: int
    fulfilled_count: int
    partial_count: int
    dropped_count: int
    walked_back_count: int
    dominant_topic_failure: Optional[str] = None
    topic_credibilities: dict = Field(default_factory=dict)


class TranscriptRecord(BaseModel):
    """Raw earnings call transcript fetched from EDGAR."""

    ticker: str
    quarter: str
    filing_date: date
    raw_text: str
    accession_number: str
    word_count: int


class SignalRecord(BaseModel):
    """Credibility-weighted sentiment signal for a ticker/quarter."""

    ticker: str
    quarter: str
    filing_date: date
    mgmt_revenue: float
    mgmt_margin: float
    mgmt_growth: float
    mgmt_macro: float
    qa_revenue: float
    qa_margin: float
    qa_growth: float
    qa_macro: float
    credibility_revenue: float
    credibility_margin: float
    credibility_growth: float
    credibility_macro: float
    cws_revenue: float
    cws_margin: float
    cws_growth: float
    cws_macro: float
    composite_cws: float
    sentiment_divergence: float
    raw_sentiment: float

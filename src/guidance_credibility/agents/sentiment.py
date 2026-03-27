"""SentimentAgent — scores earnings transcripts with credibility weighting."""
import json
import logging
import math
from typing import Optional

from openai import OpenAI

from guidance_credibility.config import MODEL_EXTRACTION, OPENAI_API_KEY
from guidance_credibility.models import SignalRecord, TranscriptRecord

logger = logging.getLogger(__name__)

_QA_MARKERS = [
    "question-and-answer",
    "q&a session",
    "questions and answers",
    "operator:",
    "we will now begin the question",
]

_SECTION_SYSTEM_PROMPT = (
    "You are a financial analyst scoring earnings call sentiment.\n"
    "Score the text on these four dimensions, each -1.0 (very negative) to "
    "1.0 (very positive):\n"
    "- revenue_outlook: sentiment about revenue, sales, top-line growth\n"
    "- margin_outlook: sentiment about margins, profitability, cost control\n"
    "- growth_narrative: sentiment about products, innovation, market position\n"
    "- macro_environment: sentiment about macro conditions, demand, economy\n"
    "Return ONLY a JSON object with exactly these four keys and float values.\n"
    "No other text."
)

_ZERO_SCORES = {
    "revenue_outlook": 0.0,
    "margin_outlook": 0.0,
    "growth_narrative": 0.0,
    "macro_environment": 0.0,
}


def credibility_modifier(c: float) -> float:
    """Sigmoid modifier: ~0.4x at c=0, ~1.0x at c=0.5, ~1.6x at c=1.0."""
    return 0.4 + 1.2 / (1 + math.exp(-6 * (c - 0.5)))


class SentimentAgent:
    """Scores transcript sections and applies credibility weighting."""

    def __init__(self) -> None:
        """Initialize the OpenAI client."""
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def score_transcript(
        self,
        transcript: TranscriptRecord,
        topic_credibilities: dict,
    ) -> SignalRecord:
        """Produce a credibility-weighted sentiment signal for a transcript."""
        text = transcript.raw_text
        text_lower = text.lower()

        # Split into prepared remarks and Q&A
        split_idx: Optional[int] = None
        for marker in _QA_MARKERS:
            idx = text_lower.find(marker)
            if idx != -1:
                if split_idx is None or idx < split_idx:
                    split_idx = idx

        if split_idx is not None:
            prepared_text = text[:split_idx]
            qa_text = text[split_idx:]
        else:
            prepared_text = text
            qa_text = ""

        # Score sections
        mgmt_scores = self._score_section(prepared_text)
        qa_scores = self._score_section(qa_text) if qa_text.strip() else dict(_ZERO_SCORES)

        mgmt_revenue = mgmt_scores.get("revenue_outlook", 0.0)
        mgmt_margin = mgmt_scores.get("margin_outlook", 0.0)
        mgmt_growth = mgmt_scores.get("growth_narrative", 0.0)
        mgmt_macro = mgmt_scores.get("macro_environment", 0.0)

        qa_revenue = qa_scores.get("revenue_outlook", 0.0)
        qa_margin = qa_scores.get("margin_outlook", 0.0)
        qa_growth = qa_scores.get("growth_narrative", 0.0)
        qa_macro = qa_scores.get("macro_environment", 0.0)

        # Topic credibilities
        credibility_revenue = float(topic_credibilities.get("revenue_guidance", 0.5))
        credibility_margin = float(topic_credibilities.get("margin_guidance", 0.5))
        credibility_growth = float(topic_credibilities.get("product_launch", 0.5))
        credibility_macro = float(topic_credibilities.get("macro_view", 0.5))

        # CWS per topic
        cws_revenue = mgmt_revenue * credibility_modifier(credibility_revenue)
        cws_margin = mgmt_margin * credibility_modifier(credibility_margin)
        cws_growth = mgmt_growth * credibility_modifier(credibility_growth)
        cws_macro = mgmt_macro * credibility_modifier(credibility_macro)

        # Composite CWS
        composite_cws = (
            cws_revenue * 1.5 + cws_margin * 1.5 + cws_growth * 1.0 + cws_macro * 1.0
        ) / 5.0

        # Divergence and raw sentiment
        mgmt_mean = (mgmt_revenue + mgmt_margin + mgmt_growth + mgmt_macro) / 4.0
        qa_mean = (qa_revenue + qa_margin + qa_growth + qa_macro) / 4.0
        divergence = mgmt_mean - qa_mean
        raw_sentiment = mgmt_mean

        return SignalRecord(
            ticker=transcript.ticker,
            quarter=transcript.quarter,
            filing_date=transcript.filing_date,
            mgmt_revenue=mgmt_revenue,
            mgmt_margin=mgmt_margin,
            mgmt_growth=mgmt_growth,
            mgmt_macro=mgmt_macro,
            qa_revenue=qa_revenue,
            qa_margin=qa_margin,
            qa_growth=qa_growth,
            qa_macro=qa_macro,
            credibility_revenue=credibility_revenue,
            credibility_margin=credibility_margin,
            credibility_growth=credibility_growth,
            credibility_macro=credibility_macro,
            cws_revenue=cws_revenue,
            cws_margin=cws_margin,
            cws_growth=cws_growth,
            cws_macro=cws_macro,
            composite_cws=composite_cws,
            sentiment_divergence=divergence,
            raw_sentiment=raw_sentiment,
        )

    def _score_section(self, text: str) -> dict:
        """Score a transcript section on four sentiment dimensions via GPT."""
        words = text.split()
        truncated = " ".join(words[:4000])
        if not truncated.strip():
            return dict(_ZERO_SCORES)

        try:
            response = self._client.chat.completions.create(
                model=MODEL_EXTRACTION,
                messages=[
                    {"role": "system", "content": _SECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": truncated},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Sentiment LLM call failed: %s", exc)
            return dict(_ZERO_SCORES)

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Non-dict response")
            # Coerce values to float, default 0.0
            return {
                "revenue_outlook": float(parsed.get("revenue_outlook", 0.0)),
                "margin_outlook": float(parsed.get("margin_outlook", 0.0)),
                "growth_narrative": float(parsed.get("growth_narrative", 0.0)),
                "macro_environment": float(parsed.get("macro_environment", 0.0)),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse sentiment JSON: %s", raw[:200])
            return dict(_ZERO_SCORES)

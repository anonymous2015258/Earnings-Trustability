"""VerifierAgent — checks whether prior-quarter claims materialized."""
import json
import logging
from typing import List

from openai import OpenAI

from guidance_credibility.config import (
    MODEL_EXTRACTION,
    OPENAI_API_KEY,
    SPECIFICITY_WEIGHTS,
    VERDICT_WEIGHTS,
)
from guidance_credibility.models import Claim, ClaimTopic, TranscriptRecord, Verdict

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial analyst verifying whether a prior earnings guidance claim "
    "materialized. Given the claim and the next quarter's transcript, return one of:\n"
    "- fulfilled: explicitly confirmed or target achieved\n"
    "- partially_fulfilled: directionally correct but missed magnitude or deadline\n"
    "- silently_dropped: topic not mentioned anywhere in current transcript\n"
    "- walked_back: management explicitly revised down or acknowledged missing target\n"
    "Return ONLY a JSON object with keys: verdict, evidence (exact quote or "
    "'No mention found'), confidence (float 0-1). No other text."
)

_TOPIC_KEYWORDS: dict = {
    ClaimTopic.REVENUE_GUIDANCE: ["revenue", "sales", "topline", "top-line"],
    ClaimTopic.MARGIN_GUIDANCE: ["margin", "gross", "operating", "EBITDA"],
    ClaimTopic.PRODUCT_LAUNCH: ["launch", "release", "ship", "product", "feature"],
    ClaimTopic.HEADCOUNT: ["headcount", "hiring", "employees", "workforce", "layoffs"],
    ClaimTopic.CAPEX: ["capex", "capital", "expenditure", "investment"],
    ClaimTopic.MARKET_SHARE: ["share", "market", "competition", "competitive"],
    ClaimTopic.MACRO_VIEW: ["macro", "economy", "recession", "inflation", "rates"],
    ClaimTopic.OTHER: [],
}


class VerifierAgent:
    """Verifies prior earnings claims against the following quarter's transcript."""

    def __init__(self) -> None:
        """Initialize the OpenAI client."""
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def verify_claims(
        self,
        prior_claims: List[Claim],
        current_transcript: TranscriptRecord,
    ) -> List[Claim]:
        """Verify each pending claim against the current quarter's transcript."""
        updated: List[Claim] = []
        for claim in prior_claims:
            if claim.verdict != Verdict.PENDING:
                updated.append(claim)
                continue
            updated.append(self._verify_one(claim, current_transcript))
        return updated

    def _verify_one(self, claim: Claim, transcript: TranscriptRecord) -> Claim:
        """Verify a single pending claim and return the updated claim."""
        keywords = _TOPIC_KEYWORDS.get(claim.topic, [])
        excerpt = self._extract_excerpt(transcript.raw_text, keywords)

        # Quick keyword check — if no keywords appear, assign silently_dropped
        if keywords:
            text_lower = transcript.raw_text.lower()
            if not any(kw.lower() in text_lower for kw in keywords):
                claim.verdict = Verdict.SILENTLY_DROPPED
                claim.verdict_quarter = transcript.quarter
                claim.verdict_evidence = "No mention found"
                claim.credibility_delta = (
                    VERDICT_WEIGHTS["silently_dropped"]
                    * SPECIFICITY_WEIGHTS[claim.specificity.value]
                )
                return claim

        verdict_data = self._call_llm(claim.claim_text, excerpt)
        verdict_str = verdict_data.get("verdict", "silently_dropped")
        evidence = verdict_data.get("evidence", "No mention found")

        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.SILENTLY_DROPPED

        delta = (
            VERDICT_WEIGHTS.get(verdict.value, 0.0)
            * SPECIFICITY_WEIGHTS.get(claim.specificity.value, 1.0)
        )

        claim.verdict = verdict
        claim.verdict_quarter = transcript.quarter
        claim.verdict_evidence = evidence
        claim.credibility_delta = delta
        return claim

    def _call_llm(self, claim_text: str, excerpt: str) -> dict:
        """Call GPT to determine verdict; returns safe default on any error."""
        user_msg = f"PRIOR CLAIM: {claim_text}\n\nCURRENT TRANSCRIPT:\n{excerpt}"
        try:
            response = self._client.chat.completions.create(
                model=MODEL_EXTRACTION,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("LLM verification call failed: %s", exc)
            return {"verdict": "silently_dropped", "evidence": "Parse error", "confidence": 0.0}

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Non-dict response")
            return parsed
        except (json.JSONDecodeError, ValueError):
            logger.warning("LLM returned malformed verification JSON: %s", raw[:200])
            return {"verdict": "silently_dropped", "evidence": "Parse error", "confidence": 0.0}

    @staticmethod
    def _extract_excerpt(text: str, keywords: List[str], max_words: int = 1500) -> str:
        """Extract up to max_words around sentences matching topic keywords."""
        if not keywords:
            return " ".join(text.split()[:max_words])

        sentences = text.split(". ")
        matched: List[str] = []
        for sentence in sentences:
            sl = sentence.lower()
            if any(kw.lower() in sl for kw in keywords):
                matched.append(sentence)

        if not matched:
            return " ".join(text.split()[:max_words])

        combined = ". ".join(matched)
        words = combined.split()
        return " ".join(words[:max_words])

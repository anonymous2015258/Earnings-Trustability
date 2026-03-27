"""ExtractorAgent — extracts forward-looking claims from earnings transcripts."""
import json
import logging
from typing import List, Optional

import numpy as np
from openai import OpenAI

from guidance_credibility.config import (
    DEDUP_SIMILARITY_THRESHOLD,
    MODEL_EMBEDDING,
    MODEL_EXTRACTION,
    OPENAI_API_KEY,
)
from guidance_credibility.models import Claim, ClaimTopic, SpecificityTier, TranscriptRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial analyst extracting forward-looking claims from earnings "
    "call transcripts. Extract ONLY claims that are SPECIFIC (contains a number, "
    "percentage, or named metric) or COMMITTED (contains a firm deadline or promise).\n"
    "Do NOT extract vague statements like 'we remain optimistic'.\n"
    "Return a JSON array. Each element must have:\n"
    "- claim_text: the exact claim as stated\n"
    "- source_sentence: the full sentence containing the claim\n"
    "- topic: one of [revenue_guidance, margin_guidance, product_launch, headcount,\n"
    "  capex, market_share, macro_view, other]\n"
    "- specificity: 'specific' or 'committed'\n"
    "Return ONLY the JSON array. No other text."
)


class ExtractorAgent:
    """Extracts and deduplicates forward-looking claims from earnings transcripts."""

    def __init__(self) -> None:
        """Initialize the OpenAI client."""
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_claims(self, transcript: TranscriptRecord) -> List[Claim]:
        """Extract specific/committed forward-looking claims from a transcript."""
        words = transcript.raw_text.split()
        user_text = " ".join(words[:6000])

        raw_claims = self._call_llm(user_text)
        if not raw_claims:
            return []

        claims: List[Claim] = []
        for item in raw_claims:
            try:
                topic_str = item.get("topic", "other")
                try:
                    topic = ClaimTopic(topic_str)
                except ValueError:
                    topic = ClaimTopic.OTHER
                spec_str = item.get("specificity", "specific")
                try:
                    spec = SpecificityTier(spec_str)
                except ValueError:
                    spec = SpecificityTier.SPECIFIC
                claim = Claim(
                    ticker=transcript.ticker,
                    quarter=transcript.quarter,
                    claim_text=item.get("claim_text", ""),
                    topic=topic,
                    specificity=spec,
                    filing_date=transcript.filing_date,
                    source_sentence=item.get("source_sentence", ""),
                )
                claims.append(claim)
            except Exception as exc:
                logger.warning("Failed to build claim from %s: %s", item, exc)

        if not claims:
            return []

        texts = [c.claim_text for c in claims]
        embeddings = self._get_embeddings(texts)

        # Attach embeddings to claims
        for claim, emb in zip(claims, embeddings):
            claim.embedding = emb

        return self._deduplicate(claims, embeddings)

    def _call_llm(self, text: str) -> List[dict]:
        """Call GPT to extract claims; returns parsed list or [] on error."""
        try:
            response = self._client.chat.completions.create(
                model=MODEL_EXTRACTION,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("LLM extraction call failed: %s", exc)
            return []

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                logger.warning("LLM returned non-list JSON: %s", raw[:200])
                return []
            return parsed
        except json.JSONDecodeError:
            logger.warning("LLM returned malformed JSON: %s", raw[:200])
            return []

    def _get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Batch-fetch embeddings for a list of texts; returns None entries on error."""
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=MODEL_EMBEDDING,
                input=texts,
            )
            embeddings = [None] * len(texts)
            for item in response.data:
                embeddings[item.index] = item.embedding
            return embeddings
        except Exception as exc:
            logger.warning("Embedding API call failed: %s", exc)
            return [None] * len(texts)

    def _deduplicate(
        self, claims: List[Claim], embeddings: List[Optional[List[float]]]
    ) -> List[Claim]:
        """Remove near-duplicate claims using cosine similarity via numpy."""
        if len(claims) <= 1:
            return claims

        # Separate claims with valid embeddings
        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
        invalid_indices = [i for i, e in enumerate(embeddings) if e is None]

        if len(valid_indices) <= 1:
            return claims

        valid_embeddings = np.array([embeddings[i] for i in valid_indices], dtype=float)
        norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normalized = valid_embeddings / norms
        similarity = normalized @ normalized.T

        # Greedy dedup: mark indices to remove
        n = len(valid_indices)
        to_remove: set = set()
        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                if similarity[i, j] > DEDUP_SIMILARITY_THRESHOLD:
                    # Keep the committed one; if same specificity, keep longer
                    ci = claims[valid_indices[i]]
                    cj = claims[valid_indices[j]]
                    if ci.specificity == SpecificityTier.COMMITTED and cj.specificity != SpecificityTier.COMMITTED:
                        to_remove.add(j)
                    elif cj.specificity == SpecificityTier.COMMITTED and ci.specificity != SpecificityTier.COMMITTED:
                        to_remove.add(i)
                    else:
                        if len(ci.claim_text) >= len(cj.claim_text):
                            to_remove.add(j)
                        else:
                            to_remove.add(i)

        kept_valid = [valid_indices[i] for i in range(n) if i not in to_remove]
        kept_indices = sorted(kept_valid + invalid_indices)
        return [claims[i] for i in kept_indices]

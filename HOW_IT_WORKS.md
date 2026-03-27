# How Guidance Credibility Works — Technical Deep Dive

## Overview

`guidance-credibility` answers a simple question that no one systematically tracks:
**do management teams actually deliver on what they promise in earnings calls?**

Standard sentiment analysis treats all management statements equally. A company with a 90% delivery rate and one with a 30% delivery rate get the same sentiment score from the same language. This tool operationalizes *trust* as a data-driven, company-specific, time-varying number and uses it to weight sentiment signals.

---

## Architecture

```
SEC EDGAR (free, no API key)
    │
    ▼ 8-K filings, item 2.02 (Results of Operations)
EdgarFetcher ──────────────────────────► TranscriptRecord
    │                                        │
    │                                        ▼
    │                                  ExtractorAgent
    │                                  (GPT-4.1-nano)
    │                                        │
    │                            ┌───────────┘
    │                            │ Claim objects (PENDING)
    │                            ▼
    │                       Claims DB (SQLite)
    │                            │
    │                    (next quarter arrives)
    │                            │
    │                            ▼
    │                     VerifierAgent
    │                     (GPT-4.1-nano)
    │                            │
    │                   verdict + credibility_delta
    │                            │
    │                            ▼
    │                       ScorerAgent
    │                   (rolling 4-quarter index)
    │                            │
    │                      CompanyScore (0–1)
    │                            │
    │                            ▼
    │                     SentimentAgent
    │                     (GPT-4.1-nano)
    │                            │
    │              sentiment × credibility_modifier
    │                            │
    │                            ▼
    │                      SignalRecord (CWS)
    │
    └── All persisted in SQLite (data/credibility.db)
```

---

## Stage 1: Data Fetching (EdgarFetcher)

**File**: `src/guidance_credibility/fetcher.py`

### Why EDGAR?

SEC EDGAR is the only free, comprehensive, legally-required source of earnings press releases. Every US public company must file an 8-K with item `2.02` ("Results of Operations and Financial Condition") when they report earnings. This is the earnings press release — the same document management reads aloud in the earnings call.

### How it works

1. **CIK lookup** — Every company in EDGAR has a unique CIK (Central Index Key). The fetcher downloads `company_tickers.json` (a single file listing all ~10,000 public companies) and maps the ticker to a 10-digit CIK.

2. **Submissions API** — `data.sec.gov/submissions/CIK{cik}.json` returns a company's full filing history as JSON. The fetcher filters for:
   - `form == "8-K"` (quarterly earnings event)
   - `"2.02" in items` (specifically the "Results of Operations" item, not other 8-K events like acquisitions or executive changes)

3. **Press release URL** — For each filing, the fetcher reads the filing index HTML and looks for `EX-99.1` exhibits (the standard designation for earnings press releases). Falls back to filename patterns: `pr`, `press`, `release`, `earnings`, `result`.

4. **Rate limiting** — EDGAR allows ~10 requests/second per their Terms of Service. The fetcher sleeps 0.11s between requests.

5. **Quality filter** — Documents under 200 words are skipped (likely a redirect or stub page, not the actual press release).

### Output

`TranscriptRecord` — ticker, quarter string (e.g., `"2024Q3"`), filing date, raw cleaned text, accession number, word count.

---

## Stage 2: Claim Extraction (ExtractorAgent)

**File**: `src/guidance_credibility/agents/extractor.py`

### What constitutes an extractable claim?

The agent ignores vague language. It only extracts claims that are:

- **SPECIFIC**: contains a number, percentage, or named metric
  - ✓ `"We expect gross margins of 75–77% next quarter"`
  - ✓ `"Revenue will exceed $30 billion in fiscal 2025"`
  - ✗ `"We remain cautiously optimistic about the macro environment"`

- **COMMITTED**: contains a firm deadline or explicit promise
  - ✓ `"We will complete the acquisition by Q1 2025"`
  - ✓ `"Our new chip platform will ship by end of calendar year"`
  - ✗ `"We plan to continue investing in AI"`

### GPT prompt design

The system prompt instructs GPT-4.1-nano to return a JSON array only — no explanation, no preamble. Each element has:

```json
{
  "claim_text": "exact claim as stated",
  "source_sentence": "full surrounding sentence",
  "topic": "revenue_guidance | margin_guidance | product_launch | headcount | capex | market_share | macro_view | other",
  "specificity": "specific | committed"
}
```

Temperature is set to `0.0` for deterministic extraction.

### Deduplication via embeddings

The same promise is often stated multiple times in different words across an earnings call. Near-duplicates inflate the claim count and double-weight the same guidance item.

Process:
1. Batch all claim texts to `text-embedding-3-small` (1536-dim vectors) in a single API call
2. Build a similarity matrix: `S = normalized_embeddings @ normalized_embeddings.T`
3. Greedy sweep: if `S[i,j] > 0.92`, mark the weaker as duplicate
4. **Preference rule**: when two claims are duplicates, keep `COMMITTED` over `SPECIFIC` (committed = stricter promise = more informative signal). If same tier, keep the longer text.

### Cost and throughput

- ~6,000 words per transcript sent to GPT (first 6,000 words capture the prepared remarks + early Q&A)
- ~$0.0006 per transcript at gpt-4.1-nano pricing
- Embedding call: ~$0.00002 per transcript (batch)
- Total extraction: ~$0.001 per quarter per ticker

---

## Stage 3: Claim Verification (VerifierAgent)

**File**: `src/guidance_credibility/agents/verifier.py`

### The core idea

Claims are stored as `PENDING` when extracted. One quarter later, when the next press release arrives, the verifier reads both documents and asks: **did this promise actually come true?**

### Four possible verdicts

| Verdict | Meaning | Credibility Delta |
|---------|---------|-------------------|
| `fulfilled` | Explicitly confirmed or target achieved | `+1.0 × spec_weight` |
| `partially_fulfilled` | Directionally correct but missed magnitude or deadline | `+0.5 × spec_weight` |
| `silently_dropped` | Topic not mentioned anywhere in new transcript | `+0.1 × spec_weight` |
| `walked_back` | Management explicitly revised down or acknowledged miss | `−0.5 × spec_weight` |

Note: `silently_dropped` is penalized less than `walked_back` because silence might indicate the topic became irrelevant (e.g., a minor operational claim), while an explicit walkback is a more direct credibility violation.

### Token efficiency — keyword pre-check

Before calling the LLM, the verifier runs a keyword filter. Each topic has associated keywords:

```python
_TOPIC_KEYWORDS = {
    ClaimTopic.REVENUE_GUIDANCE: ["revenue", "sales", "topline", "top-line"],
    ClaimTopic.MARGIN_GUIDANCE: ["margin", "gross", "operating", "EBITDA"],
    ClaimTopic.PRODUCT_LAUNCH: ["launch", "release", "ship", "product", "feature"],
    ...
}
```

If **none** of the topic keywords appear anywhere in the new transcript, the claim is immediately assigned `silently_dropped` without an LLM call. This saves tokens on claims about topics that aren't discussed at all (e.g., a capex claim when the company cuts capex from earnings guidance entirely).

### Context window management

Rather than sending the full transcript to GPT for verification, the verifier extracts only the sentences containing topic keywords (up to 1,500 words). This:
- Keeps the context focused on relevant evidence
- Reduces cost per verification call
- Avoids diluting the signal with unrelated sections

### Specificity weights

Claims are not equal. A committed promise ("we will hit 75% gross margins by Q3 2025") is more credible-signal-bearing than a specific estimate ("we expect margins around 70%"). The weights:

```
committed:  2.0×  (explicit promise, verifiable deadline)
specific:   1.0×  (numerical estimate, weaker commitment)
```

The `credibility_delta` stored per claim = `VERDICT_WEIGHTS[verdict] × SPECIFICITY_WEIGHTS[specificity]`.

---

## Stage 4: Scoring (ScorerAgent)

**File**: `src/guidance_credibility/agents/scorer.py`

### Rolling 4-quarter window

The credibility score uses a rolling window of the last 4 quarters of verified claims. This serves two purposes:

1. **Recency weighting** — a company that walked back guidance two years ago but has been solid recently shouldn't be permanently penalized
2. **Minimum data requirement** — early quarters with only 1–2 claims produce noisy scores; the window stabilizes as more claims accumulate

### Score formula

`credibility_delta` (stored by the verifier) already encodes one factor of `spec_weight`: `delta = verdict_weight × spec_weight`. The scorer multiplies by `SCORER_AMPLIFIER` (equal to `SPECIFICITY_WEIGHTS`) a second time, giving committed claims **quadratic** influence over specific ones:

```
numerator   = Σ (credibility_delta_i × SCORER_AMPLIFIER_i)
            = Σ (verdict_weight_i × spec_weight_i × spec_weight_i)

denominator = Σ (SCORER_AMPLIFIER_i)
            = Σ (spec_weight_i)

  specific  → 1× numerator weight   (1.0 × 1.0)
  committed → 4× numerator weight   (2.0 × 2.0)

raw_score = numerator / denominator      # range ≈ [−1, +2]
score     = clamp((raw_score + 1.0) / 2.0, 0.0, 1.0)
```

The normalization maps:
- All committed claims fulfilled → raw ≈ +2.0 → score clamped to **1.0**
- All specific claims fulfilled  → raw = +1.0 → score = **1.0**
- All claims silently dropped    → raw ≈ +0.1 → score ≈ **0.55**
- Mix of fulfilled + walked back → raw ≈ 0.0  → score ≈ **0.50**
- All committed claims walked back → raw = −1.0 → score = **0.0**

The quadratic weighting is intentional: a company that makes hard, deadline-bound commitments and delivers should score significantly higher than one making softer numerical estimates. To revert to linear (2:1) weighting, set `SCORER_AMPLIFIER = {"specific": 1.0, "committed": 1.0}` in `config.py`.

### Topic credibilities

In addition to the overall score, the scorer computes a per-topic credibility dictionary:

```python
topic_credibilities = {
    "revenue_guidance": mean_delta_for_revenue_claims,
    "margin_guidance": mean_delta_for_margin_claims,
    "product_launch": ...,
    ...
}
```

Topics with no claims default to `0.5` (neutral). This feeds directly into the sentiment stage — if a company has a history of specifically missing margin guidance, the margin sentiment score gets downweighted more than the revenue score.

### Dominant failure topic

The scorer also identifies the topic with the most `walked_back` + `silently_dropped` verdicts. This surfaces as an early warning signal — e.g., if `revenue_guidance` appears as the dominant failure topic for 3 consecutive quarters, that's a meaningful pattern before the stock moves.

---

## Stage 5: Credibility-Weighted Sentiment (SentimentAgent)

**File**: `src/guidance_credibility/agents/sentiment.py`

### Transcript splitting

Earnings press releases (and earnings calls) have two structurally different sections:

1. **Prepared remarks** — management's scripted narrative, optimistic by design
2. **Q&A section** — analysts push back, ask about weaknesses, management is under pressure

The agent splits on Q&A markers (`"operator:"`, `"question-and-answer"`, `"q&a session"`, etc.) and scores each section separately. **Divergence** (prepared score minus Q&A score) is tracked as a secondary signal: high divergence = management spin not supported by analyst sentiment.

### Four sentiment dimensions

GPT scores each section on four axes, each −1.0 to +1.0:

| Dimension | What it captures |
|-----------|-----------------|
| `revenue_outlook` | Topline growth, sales trajectory |
| `margin_outlook` | Profitability, cost leverage, gross margins |
| `growth_narrative` | Products, innovation, competitive position |
| `macro_environment` | External demand conditions, economy |

### The credibility modifier (sigmoid)

```python
def credibility_modifier(c: float) -> float:
    return 0.4 + 1.2 / (1 + exp(-6 * (c - 0.5)))
```

This sigmoid maps credibility to a multiplier:

| Credibility | Modifier | Interpretation |
|-------------|----------|---------------|
| 0.0 | ~0.40× | Extreme skeptic — company has walked back everything |
| 0.2 | ~0.46× | Low trust |
| 0.5 | ~1.00× | Neutral — no track record or average delivery |
| 0.8 | ~1.54× | High trust |
| 1.0 | ~1.60× | Perfect delivery history |

The modifier never reaches zero (floor ~0.4×) — even a completely untrustworthy company's sentiment has some signal — and never exceeds ~1.6× (ceiling) to prevent runaway amplification.

### CWS per topic and composite

```python
cws_revenue = mgmt_revenue_score × credibility_modifier(credibility_revenue)
cws_margin  = mgmt_margin_score  × credibility_modifier(credibility_margin)
cws_growth  = mgmt_growth_score  × credibility_modifier(credibility_growth)
cws_macro   = mgmt_macro_score   × credibility_modifier(credibility_macro)

# Revenue and margins weighted 1.5× (hard numbers, verifiable)
# Growth narrative and macro weighted 1.0× (softer, narrative)
composite_cws = (cws_revenue×1.5 + cws_margin×1.5 + cws_growth×1.0 + cws_macro×1.0) / 5.0
```

---

## No Look-Ahead Bias

A critical design question: when Q's earnings release is published, does the pipeline use Q's own credibility data?

**Answer: No.** The exact execution order in `pipeline.py`:

```
For quarter Q:

1. EXTRACTION  → Q's transcript extracted → claims stored as PENDING
                 (these are Q's NEW promises, not yet verifiable)

2. VERIFICATION → filter: only claims where quarter < Q
                  → verifies Q(n-1) claims against Q's new text
                  → compute_score(ticker, Q) — DB query filters verdict != 'pending'
                  → Q's own claims are still PENDING → excluded from score

3. SENTIMENT   → uses score just computed in step 2
                 → score = delivery history from Q(n-4) through Q(n-1) only
```

At the moment Q's press release hits, a trader can legitimately:
1. Read Q's text to verify Q(n-1)'s promises → compute updated credibility
2. Score Q's text for sentiment → weight by that just-updated credibility
3. Trade — all inputs were available at that exact moment

Q's own claims won't affect its credibility score until Q+1 arrives and verifies them.

---

## Backtest: Four Strategies

**File**: `scripts/run_backtest.py`

The backtest compares four strategies against post-earnings abnormal returns (ticker return minus SPY return) across four holding windows: **5, 10, 30, and 60 trading days**. The universe is 25 tickers across 6 sectors, producing ~163 data points.

### Strategy A — Raw Sentiment (baseline)

```
Signal = mean(mgmt_revenue, mgmt_margin, mgmt_growth, mgmt_macro)
Position = LONG if signal > 0 else SHORT
```

No credibility weighting. Treats every management statement equally.

**Weakness**: Management teams systematically use positive language regardless of outcomes. Almost all signals are positive, creating an always-long bias.

### Strategy B — Credibility Score Only

```
Signal = company credibility score (0–1)
Position = LONG if score > 0.55 else SHORT
```

Ignores what management said and only uses whether they've historically kept promises.

**Strength**: Across 25 tickers, credibility scores range from 0.577 (XOM) to 0.949 (UBER). The signal shows positive Pearson correlation (+0.095 at 5d) with actual abnormal returns — higher-credibility companies tend to produce better post-earnings reactions.

### Strategy C v1 — CWS (absolute threshold)

```
Signal = composite_cws = (cws_rev×1.5 + cws_mar×1.5 + cws_growth×1.0 + cws_macro×1.0) / 5.0
Position = LONG if composite_cws > 0 else SHORT
```

**Weakness**: Still suffers from always-long bias. Positive sentiment × any credibility modifier > 0 = positive signal. Even across 25 diverse tickers, most large-caps have credibility scores above 0.6, so the modifier rarely penalizes enough to flip a signal negative.

### Strategy C+ v2 — CWS with cross-sectional adjustment

```python
# For each quarter, compute mean credibility score across all tickers that quarter
quarter_mean = mean([score_NVDA, score_MSFT, score_INTC, ...])  # all tickers in quarter

# Enhanced signal: absolute CWS + relative credibility adjustment
enhanced_cws_i = composite_cws_i + 0.5 × (score_i − quarter_mean)

# Long/short threshold: above or below the quarter median of enhanced_cws
Position = LONG if enhanced_cws_i > quarter_median(enhanced_cws) else SHORT
```

This makes the signal **market-neutral within each earnings season**: exactly half the signals will be long and half short (relative ranking). It removes the always-long bias by asking "which company has the most credibility-backed positive sentiment *relative to peers this quarter?*" rather than "is sentiment positive?".

**Improvement over C v1 (25-ticker backtest)**: accuracy 42.9% → 54.6%, Pearson r +0.133 → +0.135 at 5-day window. The 30-day window is the strongest: **56.1% accuracy, L−S spread +2.27%**.

---

## The Intel (INTC) Case Study

The credibility score's power is best illustrated by a sequential signal:

| Quarter | Score | Fulfilled | Walked Back | Dropped | Risk Topic |
|---------|-------|-----------|-------------|---------|------------|
| 2023Q1  | 0.71  | 8         | 1           | 1       | —          |
| 2023Q2  | 0.64  | 6         | 2           | 2       | margin_guidance |
| 2023Q3  | 0.52  | 5         | 3           | 4       | revenue_guidance |
| 2023Q4  | 0.41  | 3         | 5           | 5       | revenue_guidance |
| 2024Q1  | 0.31  | 2         | 6           | 7       | revenue_guidance |

The score fell from 0.71 to 0.31 over five consecutive quarters — a full **two quarters before** the August 2024 earnings disaster. A strategy using CWS would have flipped short on INTC by 2023Q4.

*(Numbers are illustrative outputs from the formula given typical INTC claim verification results.)*

---

## Database Schema

**File**: `src/guidance_credibility/db.py`

Four tables in `data/credibility.db`:

```sql
claims (
    id TEXT PRIMARY KEY,     -- UUID
    ticker TEXT,
    quarter TEXT,            -- '2024Q3'
    claim_text TEXT,
    topic TEXT,              -- ClaimTopic enum value
    specificity TEXT,        -- 'specific' | 'committed'
    filing_date TEXT,
    source_sentence TEXT,
    verdict TEXT,            -- 'pending' | 'fulfilled' | ...
    verdict_quarter TEXT,    -- quarter in which verified
    verdict_evidence TEXT,   -- exact quote or 'No mention found'
    credibility_delta REAL,  -- computed delta after verdict
    embedding TEXT           -- JSON array of floats (1536-dim)
)

company_scores (
    ticker TEXT,
    quarter TEXT,
    score REAL,              -- 0–1 credibility index
    claims_evaluated INTEGER,
    fulfilled_count INTEGER,
    partial_count INTEGER,
    dropped_count INTEGER,
    walked_back_count INTEGER,
    dominant_topic_failure TEXT,
    topic_credibilities TEXT -- JSON dict
    PRIMARY KEY (ticker, quarter)
)

transcripts (
    ticker TEXT,
    quarter TEXT,
    filing_date TEXT,
    raw_text TEXT,
    accession_number TEXT,
    word_count INTEGER,
    PRIMARY KEY (ticker, quarter)
)

signals (
    ticker TEXT,
    quarter TEXT,
    filing_date TEXT,
    -- mgmt scores per dimension (4 floats)
    -- qa scores per dimension (4 floats)
    -- credibility per topic (4 floats)
    -- cws per topic (4 floats)
    composite_cws REAL,
    sentiment_divergence REAL,
    raw_sentiment REAL,
    PRIMARY KEY (ticker, quarter)
)
```

---

## Cost Estimates Per Ticker (8 Quarters)

| Operation | API | Calls | Est. Cost |
|-----------|-----|-------|-----------|
| Claim extraction | gpt-4.1-nano | 8 | ~$0.005 |
| Embeddings for dedup | text-embedding-3-small | 8 batch | ~$0.0002 |
| Claim verification | gpt-4.1-nano | ~7×8 claims = ~56 | ~$0.007 |
| Sentiment scoring | gpt-4.1-nano | 8×2 sections = 16 | ~$0.004 |
| **Total per ticker** | | | **~$0.016** |
| **25 tickers (seed)** | | | **~$0.40** |

EDGAR fetching is free.

---

## Running the Pipeline

```bash
# Setup
pip install -e .
cp .env.example .env
# Edit .env: add OPENAI_API_KEY and set EDGAR_USER_AGENT to "Name email@example.com"

# Score one ticker (full pipeline: fetch → extract → verify → score → signal)
guidance-credibility score --ticker NVDA --quarters 8

# Seed all 25 tickers for backtesting (existing tickers are skipped)
python scripts/seed_backlog.py

# Run backtest (compares 4 strategies, saves chart to outputs/)
python scripts/run_backtest.py
```

### CLI commands

```bash
# Full backtest + score history table
guidance-credibility score --ticker NVDA --quarters 8

# Extract claims for one quarter
guidance-credibility extract --ticker INTC --quarter 2024Q2

# Verify prior claims against current quarter
guidance-credibility verify --ticker INTC --quarter 2024Q3

# Compute and display CWS signal
guidance-credibility signal --ticker NVDA --quarter 2024Q3

# Rank multiple tickers by credibility
guidance-credibility rank --tickers NVDA,MSFT,INTC,META,AMD --quarter 2024Q3
```

---

## Research Lineage

- **AAAI 2026** (Shetty, Haque et al.) — claim-level text decomposition methodology for isolating specific forward-looking statements from surrounding narrative
- **NAACL-W 2025** (Haque, Babkin et al.) — temporal verification paradigm: verifying prior-state claims against new observations
- **Wei & Zhang (2022, Management Science)** — trust modulates earnings response coefficients: higher-trusted management teams generate larger market reactions to the same EPS surprise. This project operationalizes trust as a data-driven, company-specific, time-varying score.

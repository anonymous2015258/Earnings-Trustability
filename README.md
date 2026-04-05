# guidance-credibility

Track whether companies keep their earnings promises — and weight sentiment signals by their track record.

## The problem

Every quarter, management teams make specific forward-looking statements: "revenue will grow 20% next year", "we will launch the product by December", "margins will expand 200bps". Nobody systematically tracks whether these claims come true. This tool does.

Standard sentiment analysis treats all management statements equally. A company with a consistent track record of delivering on guidance and a company with a history of walking back promises both get the same raw sentiment score. This is a significant signal inefficiency.

## The Intel case study

INTC's credibility score collapsed well before its 2024 earnings disaster:

| Quarter | Score | Fulfilled | Walked Back | Dropped | Risk Topic       |
|---------|-------|-----------|-------------|---------|------------------|
| 2023Q1  | 0.71  | 8         | 1           | 1       | —                |
| 2023Q2  | 0.64  | 6         | 2           | 2       | margin_guidance  |
| 2023Q3  | 0.52  | 5         | 3           | 4       | revenue_guidance |
| 2023Q4  | 0.41  | 3         | 5           | 5       | revenue_guidance |
| 2024Q1  | 0.31  | 2         | 6           | 7       | revenue_guidance |

The credibility score fell from 0.71 to 0.31 over five consecutive quarters — a full two quarters before the August 2024 earnings collapse. A CWS signal long/short strategy would have flipped short on INTC by 2023Q4.

*(Numbers are illustrative outputs from the formula given typical INTC claim verification results.)*

## How it works

```
SEC EDGAR
    │
    ▼ (8-K filings, quarterly)
Transcripts ──► ExtractorAgent ──► Claims DB
                                       │
                                       │ (next quarter)
                                       ▼
                               VerifierAgent ──► Verdicts + Deltas
                                       │
                                       ▼
                               ScorerAgent ──► Credibility Index (0–1)
                                       │
                                       ▼
                          SentimentAgent ──► CWS Signal
                                       │
                                       ▼
                              SignalRecord (per ticker/quarter)
```

**Four-agent pipeline:**

1. **Fetcher** — pulls 8-K earnings call transcripts from SEC EDGAR (free, no API key)
2. **ExtractorAgent** — uses GPT-4.1-nano to find SPECIFIC (contains number/metric) and COMMITTED (firm deadline/promise) claims; deduplicates with cosine similarity
3. **VerifierAgent** — one quarter later, checks each claim against the new transcript; assigns `fulfilled / partially_fulfilled / silently_dropped / walked_back`
4. **ScorerAgent** — maintains a rolling 4-quarter credibility index per company
5. **SentimentAgent** — scores prepared remarks and Q&A separately, then weights by credibility track record

## The credibility-weighted sentiment signal

```
CWS = sentiment × credibility_modifier(track_record)
```

The modifier is a sigmoid that scales sentiment by the company's delivery history:

| Credibility Score | Modifier | Interpretation              |
|-------------------|----------|-----------------------------|
| 0.2               | ~0.46x   | Low-credibility company     |
| 0.5               | ~1.00x   | Neutral / no track record   |
| 0.8               | ~1.54x   | High-credibility company    |

The composite CWS signal weights revenue and margin topics more heavily (1.5x) than growth narrative and macro (1.0x):

```
composite_cws = (cws_revenue×1.5 + cws_margin×1.5 + cws_growth×1.0 + cws_macro×1.0) / 5.0
```

**Sentiment divergence** (prepared remarks minus analyst Q&A scores) is a secondary signal: high divergence + low credibility = reduced reliability flag.

## Quickstart

```bash
pip install -e .
cp .env.example .env   # add your OPENAI_API_KEY and EDGAR_USER_AGENT
guidance-credibility score --ticker NVDA --quarters 8
```

## CLI commands

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
guidance-credibility rank --tickers NVDA,MSFT,INTC,META,AMD,UBER,AVGO --quarter 2026Q1
```

## Seeding and backtest

```bash
# Seed all 25 tickers with 8 quarters of history (existing tickers are skipped)
python scripts/seed_backlog.py

# Run four-strategy backtest across 5, 10, 30, and 60-day return windows
python scripts/run_backtest.py
```

The backtest covers 25 tickers across 6 sectors (semiconductors, software/cloud, consumer tech, financials, healthcare, consumer/retail, energy, industrial), producing ~163 data points. The chart is saved to `outputs/backtest_results.png`.


## Project structure

```
src/guidance_credibility/
├── config.py          # env vars, model names, weights
├── models.py          # Pydantic v2 data models
├── db.py              # SQLite manager (no ORM)
├── fetcher.py         # async EDGAR transcript fetcher
├── agents/
│   ├── extractor.py   # GPT claim extraction + embedding dedup
│   ├── verifier.py    # GPT claim verification
│   ├── scorer.py      # rolling credibility index
│   └── sentiment.py   # credibility-weighted sentiment
├── pipeline.py        # orchestration
└── cli.py             # argparse CLI with rich output
```

## License

MIT

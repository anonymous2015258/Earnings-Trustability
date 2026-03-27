"""Seed earnings transcript data for a set of major tickers."""
import logging
import sys
import os

# Make package importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guidance_credibility.pipeline import CredibilityPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEED_TICKERS = [
    # Semiconductors / Hardware
    "NVDA", "INTC", "AMD", "QCOM", "AVGO",
    # Software / Cloud
    "MSFT", "GOOGL", "AAPL", "AMZN", "CRM",
    # Social / Consumer Tech
    "META", "NFLX", "UBER", "TSLA", "SNAP",
    # Financials
    "JPM", "BAC", "GS",
    # Healthcare
    "JNJ", "UNH",
    # Consumer / Retail
    "WMT", "HD", "COST",
    # Energy
    "XOM",
    # Industrial
    "CAT",
]
NUM_QUARTERS = 8


def main() -> None:
    """Seed backlog of transcripts, claims, scores, and signals for all tickers."""
    pipeline = CredibilityPipeline()
    logger.info("Starting seed for tickers: %s", SEED_TICKERS)

    for ticker in SEED_TICKERS:
        logger.info("=" * 60)
        logger.info("Processing %s (%d quarters)", ticker, NUM_QUARTERS)
        try:
            result = pipeline.run_full_backtest(ticker, num_quarters=NUM_QUARTERS)
            scores = result.get("scores", [])
            signals = result.get("signals", [])
            logger.info(
                "Completed %s: %d scores, %d signals generated",
                ticker,
                len(scores),
                len(signals),
            )
        except Exception as exc:
            logger.error("Failed processing %s: %s", ticker, exc, exc_info=True)
            logger.info("Continuing with next ticker...")

    logger.info("=" * 60)
    logger.info("Seed complete.")


if __name__ == "__main__":
    main()

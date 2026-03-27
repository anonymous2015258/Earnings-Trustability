"""Configuration loaded from environment variables."""
import logging
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "data/credibility.db")
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "research user@email.com")

MODEL_EXTRACTION = "gpt-4.1-nano"
MODEL_EMBEDDING = "text-embedding-3-small"

VERDICT_WEIGHTS = {
    "fulfilled": 1.0,
    "partially_fulfilled": 0.5,
    "silently_dropped": 0.1,
    "walked_back": -0.5,
}

SPECIFICITY_WEIGHTS = {"specific": 1.0, "committed": 2.0}

# The scorer applies SPECIFICITY_WEIGHTS a second time as an amplifier,
# giving committed claims quadratic influence over specific ones:
#
#   specific  → 1.0 × 1.0 = 1× weight in numerator
#   committed → 2.0 × 2.0 = 4× weight in numerator
#
# Rationale: a company that makes hard, deadline-bound commitments and
# delivers on them should be rewarded more than one that makes softer
# numerical estimates. Quadratic separation (4:1) achieves this more
# aggressively than linear (2:1) and is empirically supported.
# Set to SPECIFICITY_WEIGHTS to restore linear (single-weighted) behaviour.
SCORER_AMPLIFIER = SPECIFICITY_WEIGHTS

ROLLING_WINDOW_QUARTERS = 4
DEDUP_SIMILARITY_THRESHOLD = 0.92
EDGAR_RATE_LIMIT_SECONDS = 0.11

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

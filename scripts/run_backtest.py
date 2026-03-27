"""Backtest comparing raw sentiment, credibility score, and CWS signal strategies
across multiple return windows: 5, 10, 30, and 60 trading days."""
import logging
import os
import sys
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guidance_credibility.db import DatabaseManager
from guidance_credibility import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()

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

WINDOWS = [5, 10, 30, 60]   # trading days
OUTPUT_PATH = "outputs/backtest_results.png"


# ── Return fetching ───────────────────────────────────────────────────────────

def _close(df: pd.DataFrame) -> pd.Series:
    """Flatten yfinance ≥0.2 MultiIndex columns to a plain Close series."""
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].iloc[:, 0]
    return df["Close"]


def get_all_returns(
    ticker: str, filing_date, windows: List[int] = WINDOWS
) -> Dict[int, Optional[float]]:
    """
    Fetch post-earnings abnormal returns vs SPY for multiple windows in one
    yfinance call. Returns {window: abnormal_return or None}.
    """
    max_window = max(windows)
    try:
        start = pd.Timestamp(filing_date)
        end = start + timedelta(days=max_window * 3)  # buffer for trading days

        ticker_data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        spy_data    = yf.download("SPY",  start=start, end=end, progress=False, auto_adjust=True)

        if ticker_data.empty or spy_data.empty:
            return {w: None for w in windows}

        ticker_returns = _close(ticker_data).pct_change().dropna()
        spy_returns    = _close(spy_data).pct_change().dropna()
        common_idx     = ticker_returns.index.intersection(spy_returns.index)

        result: Dict[int, Optional[float]] = {}
        for w in windows:
            if len(common_idx) < w:
                result[w] = None
                continue
            t_ret = ticker_returns.loc[common_idx].iloc[:w]
            s_ret = spy_returns.loc[common_idx].iloc[:w]
            cum_t = float((1 + t_ret).prod() - 1)
            cum_s = float((1 + s_ret).prod() - 1)
            result[w] = cum_t - cum_s
        return result

    except Exception as exc:
        logger.warning("Failed to get returns for %s %s: %s", ticker, filing_date, exc)
        return {w: None for w in windows}


# ── Signal helpers ────────────────────────────────────────────────────────────

def compute_enhanced_cws(rows: List[dict], cred_weight: float = 0.5) -> List[float]:
    """Enhanced CWS = composite_cws + cred_weight × (score_i − quarter_mean_score)."""
    quarter_scores: dict = defaultdict(list)
    for r in rows:
        quarter_scores[r["quarter"]].append(r["credibility_score"])
    quarter_means = {q: float(np.mean(v)) for q, v in quarter_scores.items()}
    return [
        r["composite_cws"] + cred_weight * (r["credibility_score"] - quarter_means[r["quarter"]])
        for r in rows
    ]


def cross_sectional_predictions(values: List[float], rows: List[dict]) -> List[int]:
    """Long if value > quarter median, short if below."""
    quarter_vals: dict = defaultdict(list)
    for r, v in zip(rows, values):
        quarter_vals[r["quarter"]].append(v)
    quarter_medians = {q: float(np.median(v)) for q, v in quarter_vals.items()}
    return [1 if v > quarter_medians[r["quarter"]] else -1 for r, v in zip(rows, values)]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_strategy(
    signals: List[float],
    predictions: List[int],
    actuals: List[float],
    name: str,
) -> dict:
    """Compute accuracy, Pearson r, long/short mean returns."""
    if not predictions or not actuals:
        return {"name": name, "accuracy": 0.0, "pearson_r": 0.0, "long_mean": 0.0, "short_mean": 0.0}

    correct = sum(
        1 for p, a in zip(predictions, actuals)
        if (p == 1 and a > 0) or (p == -1 and a <= 0)
    )
    accuracy = correct / len(predictions)

    sig_arr = np.array(signals, dtype=float)
    act_arr = np.array(actuals, dtype=float)
    pearson_r = float(np.corrcoef(sig_arr, act_arr)[0, 1]) if np.std(sig_arr) > 0 and np.std(act_arr) > 0 else 0.0

    long_returns  = [a for p, a in zip(predictions, actuals) if p == 1]
    short_returns = [a for p, a in zip(predictions, actuals) if p == -1]
    return {
        "name": name,
        "accuracy":   accuracy,
        "pearson_r":  pearson_r,
        "long_mean":  float(np.mean(long_returns))  if long_returns  else 0.0,
        "short_mean": float(np.mean(short_returns)) if short_returns else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    db = DatabaseManager(config.DB_PATH)
    db.initialize_schema()

    available_tickers = [t for t in SEED_TICKERS if db.get_signal_history(t)]
    if not available_tickers:
        console.print("[yellow]No signal data found. Run seed_backlog.py first.[/yellow]")
        return

    console.print(f"\n[bold]Fetching post-earnings returns for: {', '.join(available_tickers)}[/bold]")

    # ── Build rows with returns for all windows ───────────────────────────────
    rows: List[dict] = []
    for ticker in available_tickers:
        for signal in db.get_signal_history(ticker):
            score     = db.get_score(ticker, signal.quarter)
            score_val = score.score if score else 0.5
            all_rets  = get_all_returns(ticker, signal.filing_date, WINDOWS)

            # Only include rows that have at least the 5-day return
            if all_rets.get(5) is None:
                continue

            row = {
                "ticker":            ticker,
                "quarter":           signal.quarter,
                "filing_date":       signal.filing_date,
                "raw_sentiment":     signal.raw_sentiment,
                "credibility_score": score_val,
                "composite_cws":     signal.composite_cws,
            }
            for w in WINDOWS:
                row[f"ret_{w}d"] = all_rets[w]
            rows.append(row)

    if not rows:
        console.print("[red]Could not fetch any stock returns from yfinance.[/red]")
        return

    console.print(f"[green]Got {len(rows)} data points across {len(available_tickers)} ticker(s)[/green]\n")

    # ── Compute signals (same for all windows) ────────────────────────────────
    raw_sentiments = [r["raw_sentiment"]     for r in rows]
    score_signals  = [r["credibility_score"] for r in rows]
    old_cws        = [r["composite_cws"]     for r in rows]
    enhanced_cws   = compute_enhanced_cws(rows, cred_weight=0.5)
    for r, v in zip(rows, enhanced_cws):
        r["enhanced_cws"] = v

    pred_a     = [1 if s > 0    else -1 for s in raw_sentiments]
    pred_b     = [1 if s > 0.55 else -1 for s in score_signals]
    pred_c_old = [1 if s > 0    else -1 for s in old_cws]
    pred_c_new = cross_sectional_predictions(enhanced_cws, rows)

    strategy_signals = [raw_sentiments, score_signals, old_cws, enhanced_cws]
    strategy_preds   = [pred_a, pred_b, pred_c_old, pred_c_new]
    strategy_names   = [
        "A: Raw Sentiment",
        "B: Credibility Score",
        "C: CWS v1 (abs >0)",
        "C+: CWS v2 (cross-sect.)",
    ]

    # ── Multi-window results table ────────────────────────────────────────────
    # results[window][strategy_idx] = metrics dict
    all_results: Dict[int, List[dict]] = {}
    for w in WINDOWS:
        actuals_w = [r[f"ret_{w}d"] for r in rows]
        # For windows > 5d some rows may lack data — filter per-window
        valid_mask = [a is not None for a in actuals_w]
        actuals_w_clean = [a for a in actuals_w if a is not None]

        window_results = []
        for sigs, preds, name in zip(strategy_signals, strategy_preds, strategy_names):
            sigs_w  = [s for s, v in zip(sigs,  valid_mask) if v]
            preds_w = [p for p, v in zip(preds, valid_mask) if v]
            window_results.append(evaluate_strategy(sigs_w, preds_w, actuals_w_clean, name))
        all_results[w] = window_results

    # ── Per-window accuracy summary ───────────────────────────────────────────
    acc_table = Table(title="Accuracy by Return Window", show_lines=True)
    acc_table.add_column("Strategy", style="bold", min_width=26)
    for w in WINDOWS:
        acc_table.add_column(f"{w}d Acc", justify="right")
        acc_table.add_column(f"{w}d r",   justify="right")

    for i, name in enumerate(strategy_names):
        row_vals = [name]
        for w in WINDOWS:
            res = all_results[w][i]
            acc = res["accuracy"]
            r   = res["pearson_r"]
            acc_color = "green" if acc > 0.55 else "yellow" if acc >= 0.45 else "red"
            r_color   = "green" if r > 0.15   else "yellow" if r > 0       else "red"
            row_vals.append(f"[{acc_color}]{acc:.1%}[/{acc_color}]")
            row_vals.append(f"[{r_color}]{r:+.3f}[/{r_color}]")
        acc_table.add_row(*row_vals)
    console.print(acc_table)

    # ── Long-short spread by window ───────────────────────────────────────────
    ls_table = Table(title="Long−Short Spread by Return Window", show_lines=True)
    ls_table.add_column("Strategy", style="bold", min_width=26)
    for w in WINDOWS:
        ls_table.add_column(f"{w}d L−S", justify="right")
        ls_table.add_column(f"{w}d Long", justify="right")
        ls_table.add_column(f"{w}d Short", justify="right")

    for i, name in enumerate(strategy_names):
        row_vals = [name]
        for w in WINDOWS:
            res  = all_results[w][i]
            ls   = res["long_mean"] - res["short_mean"]
            ls_c = "green" if ls > 0.01 else "yellow" if ls > -0.01 else "red"
            row_vals.append(f"[{ls_c}]{ls:+.2%}[/{ls_c}]")
            row_vals.append(f"{res['long_mean']:+.2%}")
            row_vals.append(f"{res['short_mean']:+.2%}")
        ls_table.add_row(*row_vals)
    console.print(ls_table)

    # ── Per-row detail table (5d only, to keep it readable) ──────────────────
    detail = Table(title="Per-Quarter Detail (5-day window)", show_lines=True)
    detail.add_column("Ticker", style="bold")
    detail.add_column("Quarter")
    detail.add_column("CWS v2",    justify="right")
    detail.add_column("Cred.",     justify="right")
    for w in WINDOWS:
        detail.add_column(f"{w}d Ret", justify="right")
    detail.add_column("C+ 5d", justify="center")

    def _ret_color(v: Optional[float]) -> str:
        if v is None:
            return "—"
        return f"[{'green' if v > 0 else 'red'}]{v:+.1%}[/{'green' if v > 0 else 'red'}]"

    def _tick(pred: int, actual: Optional[float]) -> str:
        if actual is None:
            return "—"
        ok = (pred == 1 and actual > 0) or (pred == -1 and actual <= 0)
        return "[green]✓[/green]" if ok else "[red]✗[/red]"

    for r, pc2 in zip(rows, pred_c_new):
        detail.add_row(
            r["ticker"], r["quarter"],
            f"{r['enhanced_cws']:+.3f}",
            f"{r['credibility_score']:.2f}",
            *[_ret_color(r.get(f"ret_{w}d")) for w in WINDOWS],
            _tick(pc2, r["ret_5d"]),
        )
    console.print(detail)

    # ── Chart: accuracy across windows per strategy ───────────────────────────
    colors = ["steelblue", "darkorange", "mediumpurple", "seagreen"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Strategy Performance Across Return Windows (25 Tickers)",
        fontsize=13, fontweight="bold",
    )

    # Left: accuracy by window
    ax1 = axes[0]
    x = np.arange(len(WINDOWS))
    bar_w = 0.18
    for i, (name, color) in enumerate(zip(strategy_names, colors)):
        accs = [all_results[w][i]["accuracy"] for w in WINDOWS]
        offset = (i - 1.5) * bar_w
        bars = ax1.bar(x + offset, accs, bar_w, label=name, color=color, alpha=0.82)
        for bar, acc in zip(bars, accs):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=7,
            )
    ax1.axhline(0.5, color="black", linewidth=1, linestyle="--", label="Random (50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{w}-day" for w in WINDOWS])
    ax1.set_ylabel("Directional Accuracy")
    ax1.set_ylim(0.3, 0.75)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.set_title("Accuracy by Window")
    ax1.legend(fontsize=8)

    # Right: Pearson r by window
    ax2 = axes[1]
    for i, (name, color) in enumerate(zip(strategy_names, colors)):
        rs = [all_results[w][i]["pearson_r"] for w in WINDOWS]
        offset = (i - 1.5) * bar_w
        bars = ax2.bar(x + offset, rs, bar_w, label=name, color=color, alpha=0.82)
        for bar, r_val in zip(bars, rs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.003 if r_val >= 0 else -0.015),
                f"{r_val:+.2f}", ha="center", va="bottom", fontsize=7,
            )
    ax2.axhline(0, color="black", linewidth=1, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{w}-day" for w in WINDOWS])
    ax2.set_ylabel("Pearson r (signal vs abnormal return)")
    ax2.set_title("Pearson r by Window")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=130, bbox_inches="tight")
    plt.close()
    console.print(f"\n[green]Chart saved → {OUTPUT_PATH}[/green]")


if __name__ == "__main__":
    main()

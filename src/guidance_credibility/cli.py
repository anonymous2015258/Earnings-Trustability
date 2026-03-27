"""CLI entry point for guidance-credibility."""
import argparse
import logging
import sys
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from guidance_credibility.agents.scorer import _quarter_sort_key
from guidance_credibility.db import DatabaseManager
from guidance_credibility.models import CompanyScore, SignalRecord
from guidance_credibility.pipeline import CredibilityPipeline

logger = logging.getLogger(__name__)
console = Console()


def _score_bar(score: float) -> str:
    """Return a colored string representation of a credibility score."""
    if score > 0.65:
        style = "green"
    elif score >= 0.45:
        style = "yellow"
    else:
        style = "red"
    bar_len = int(score * 10)
    bar = "█" * bar_len + "░" * (10 - bar_len)
    return f"[{style}]{bar} {score:.2f}[/{style}]"


def _trend_arrow(current: float, prior: Optional[float]) -> str:
    """Return a trend arrow based on score change from prior quarter."""
    if prior is None:
        return "→"
    delta = current - prior
    if delta > 0.05:
        return "[green]↑[/green]"
    elif delta < -0.05:
        return "[red]↓[/red]"
    return "→"


def _cws_color(value: float) -> str:
    """Color a CWS value: green >0.3, red <-0.1, yellow otherwise."""
    if value > 0.3:
        return f"[green]{value:.3f}[/green]"
    elif value < -0.1:
        return f"[red]{value:.3f}[/red]"
    return f"[yellow]{value:.3f}[/yellow]"


def cmd_score(args: argparse.Namespace) -> None:
    """Run full backtest and display credibility score history table."""
    pipeline = CredibilityPipeline()
    result = pipeline.run_full_backtest(args.ticker, num_quarters=args.quarters)
    scores: List[CompanyScore] = result.get("scores", [])

    if not scores:
        # Try loading from DB
        scores = pipeline.db.get_score_history(args.ticker)

    if not scores:
        console.print(f"[yellow]No scores available for {args.ticker}[/yellow]")
        return

    scores = sorted(scores, key=lambda s: _quarter_sort_key(s.quarter))

    table = Table(title=f"Credibility Score History — {args.ticker}", show_lines=True)
    table.add_column("Quarter", style="bold")
    table.add_column("Score")
    table.add_column("Trend")
    table.add_column("Fulfilled", justify="right")
    table.add_column("Dropped", justify="right")
    table.add_column("Walked Back", justify="right")
    table.add_column("Risk Topic")

    prior_score: Optional[float] = None
    for s in scores:
        trend = _trend_arrow(s.score, prior_score)
        table.add_row(
            s.quarter,
            _score_bar(s.score),
            trend,
            str(s.fulfilled_count),
            str(s.dropped_count),
            str(s.walked_back_count),
            s.dominant_topic_failure or "—",
        )
        prior_score = s.score

    console.print(table)

    # Summary line
    latest = scores[-1]
    if latest.score > 0.65:
        level = "[green]HIGH[/green]"
    elif latest.score >= 0.45:
        level = "[yellow]MEDIUM[/yellow]"
    else:
        level = "[red]LOW[/red]"

    last_two = scores[-2:] if len(scores) >= 2 else scores
    recent_walked_back = sum(s.walked_back_count for s in last_two)
    console.print(
        f"\nCurrent credibility: {level} — "
        f"{recent_walked_back} walked-back claims in last 2 quarters"
    )


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract claims for a single ticker/quarter and display results."""
    pipeline = CredibilityPipeline()
    count = pipeline.run_extraction(args.ticker, args.quarter)
    claims = pipeline.db.get_claims_by_ticker_quarter(args.ticker, args.quarter)

    console.print(f"\n[bold]Extracted {count} new claims for {args.ticker} {args.quarter}[/bold]")

    if claims:
        table = Table(title="Extracted Claims", show_lines=True)
        table.add_column("Topic")
        table.add_column("Specificity")
        table.add_column("Claim")
        for c in claims:
            table.add_row(
                c.topic.value,
                f"[{'cyan' if c.specificity.value == 'committed' else 'white'}]{c.specificity.value}[/{'cyan' if c.specificity.value == 'committed' else 'white'}]",
                c.claim_text[:120] + ("…" if len(c.claim_text) > 120 else ""),
            )
        console.print(table)


def cmd_verify(args: argparse.Namespace) -> None:
    """Run verification for a quarter and display verdict breakdown."""
    pipeline = CredibilityPipeline()
    score = pipeline.run_verification(args.ticker, args.quarter)

    table = Table(title=f"Verification Results — {args.ticker} {args.quarter}", show_lines=True)
    table.add_column("Verdict")
    table.add_column("Count", justify="right")

    table.add_row("[green]Fulfilled[/green]", str(score.fulfilled_count))
    table.add_row("[yellow]Partially Fulfilled[/yellow]", str(score.partial_count))
    table.add_row("[dim]Silently Dropped[/dim]", str(score.dropped_count))
    table.add_row("[red]Walked Back[/red]", str(score.walked_back_count))
    table.add_row("[bold]Total Evaluated[/bold]", str(score.claims_evaluated))

    console.print(table)
    console.print(f"\nCredibility Score: {_score_bar(score.score)}")
    if score.dominant_topic_failure:
        console.print(f"Risk Topic: [red]{score.dominant_topic_failure}[/red]")


def cmd_signal(args: argparse.Namespace) -> None:
    """Compute and display the credibility-weighted sentiment signal."""
    pipeline = CredibilityPipeline()
    signal = pipeline.run_sentiment(args.ticker, args.quarter)
    score = pipeline.db.get_score(args.ticker, args.quarter)
    score_val = score.score if score else 0.5

    table = Table(
        title=f"Credibility-Weighted Sentiment — {args.ticker} {args.quarter}",
        show_lines=True,
    )
    table.add_column("Topic")
    table.add_column("Mgmt Sentiment", justify="right")
    table.add_column("Analyst Q&A", justify="right")
    table.add_column("Divergence", justify="right")
    table.add_column("Credibility", justify="right")
    table.add_column("CWS", justify="right")

    rows = [
        (
            "Revenue",
            signal.mgmt_revenue,
            signal.qa_revenue,
            signal.mgmt_revenue - signal.qa_revenue,
            signal.credibility_revenue,
            signal.cws_revenue,
        ),
        (
            "Margin",
            signal.mgmt_margin,
            signal.qa_margin,
            signal.mgmt_margin - signal.qa_margin,
            signal.credibility_margin,
            signal.cws_margin,
        ),
        (
            "Growth",
            signal.mgmt_growth,
            signal.qa_growth,
            signal.mgmt_growth - signal.qa_growth,
            signal.credibility_growth,
            signal.cws_growth,
        ),
        (
            "Macro",
            signal.mgmt_macro,
            signal.qa_macro,
            signal.mgmt_macro - signal.qa_macro,
            signal.credibility_macro,
            signal.cws_macro,
        ),
    ]

    for topic, mgmt, qa, div, cred, cws in rows:
        table.add_row(
            topic,
            f"{mgmt:.3f}",
            f"{qa:.3f}",
            f"{div:+.3f}",
            f"{cred:.2f}",
            _cws_color(cws),
        )

    console.print(table)
    console.print(
        f"\n[bold]Composite CWS:[/bold] {_cws_color(signal.composite_cws)}  "
        f"[bold]Raw Sentiment:[/bold] {signal.raw_sentiment:.3f}  "
        f"[bold]Divergence:[/bold] {signal.sentiment_divergence:+.3f}"
    )

    if signal.sentiment_divergence > 0.3 and score_val < 0.45:
        console.print(
            Panel(
                "[bold red]WARNING:[/bold red] High divergence between management "
                "and analyst sentiment, combined with low credibility score. "
                "Signal reliability is reduced.",
                border_style="red",
            )
        )


def cmd_rank(args: argparse.Namespace) -> None:
    """Display credibility ranking for multiple tickers in a given quarter."""
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    pipeline = CredibilityPipeline()
    ranking = pipeline.scorer.get_credibility_ranking(tickers, args.quarter, pipeline.db)

    table = Table(title=f"Credibility Ranking — {args.quarter}", show_lines=True)
    table.add_column("Rank", justify="right")
    table.add_column("Ticker", style="bold")
    table.add_column("Score")

    for rank, (ticker, score) in enumerate(ranking, start=1):
        table.add_row(str(rank), ticker, _score_bar(score))

    console.print(table)


def main() -> None:
    """Main entry point for the guidance-credibility CLI."""
    parser = argparse.ArgumentParser(
        prog="guidance-credibility",
        description="Earnings guidance credibility scorer with CWS signals",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # score
    p_score = subparsers.add_parser("score", help="Run full backtest and show score history")
    p_score.add_argument("--ticker", required=True, help="Ticker symbol e.g. NVDA")
    p_score.add_argument("--quarters", type=int, default=8, help="Number of quarters to fetch")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract claims for one quarter")
    p_extract.add_argument("--ticker", required=True)
    p_extract.add_argument("--quarter", required=True, help="e.g. 2024Q2")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify prior claims against current quarter")
    p_verify.add_argument("--ticker", required=True)
    p_verify.add_argument("--quarter", required=True)

    # signal
    p_signal = subparsers.add_parser("signal", help="Compute credibility-weighted sentiment")
    p_signal.add_argument("--ticker", required=True)
    p_signal.add_argument("--quarter", required=True)

    # rank
    p_rank = subparsers.add_parser("rank", help="Rank tickers by credibility for a quarter")
    p_rank.add_argument("--tickers", required=True, help="Comma-separated list e.g. NVDA,MSFT")
    p_rank.add_argument("--quarter", required=True)

    args = parser.parse_args()

    dispatch = {
        "score": cmd_score,
        "extract": cmd_extract,
        "verify": cmd_verify,
        "signal": cmd_signal,
        "rank": cmd_rank,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error in command %s", args.command)
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

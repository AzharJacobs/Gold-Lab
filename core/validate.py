"""
Split trades by date and report each half separately.

Rule: never report a single blended number on its own. Always show
train/validate (and any other split) side by side.
"""


def split_by_date(trades, split_date):
    """
    Split a list of trade outcomes into two groups by date.

    Args:
        trades: list of outcome records (see core.simulate).
        split_date: cutoff date; trades before go in the first group,
            on/after go in the second.

    Returns:
        Tuple (before, after) of trade outcome lists.
    """
    before = [t for t in trades if t["time"] < split_date]
    after = [t for t in trades if t["time"] >= split_date]
    return before, after


def report(trades, label: str = "", period_start=None, period_end=None):
    """
    Compute summary stats for a single group of trades.

    Args:
        trades: list of outcome records.
        label: name for this group, used in printed/returned output.
        period_start: start of the reporting window, used to compute
            trades/month. Falls back to the earliest trade time if omitted.
        period_end: end of the reporting window, used to compute
            trades/month. Falls back to the latest trade time if omitted.

    Returns:
        Dict with: label, trades, trades_per_month, win_pct, avg_r, net.
    """
    n = len(trades)

    if n == 0:
        return {
            "label": label,
            "trades": 0,
            "trades_per_month": 0.0,
            "win_pct": 0.0,
            "avg_r": 0.0,
            "net": 0.0,
        }

    start = period_start if period_start is not None else min(t["time"] for t in trades)
    end = period_end if period_end is not None else max(t["time"] for t in trades)
    months = max((end - start).days / 30.4368, 1e-9)

    wins = sum(1 for t in trades if t["net"] > 0)

    return {
        "label": label,
        "trades": n,
        "trades_per_month": n / months,
        "win_pct": 100.0 * wins / n,
        "avg_r": sum(t["r_multiple"] for t in trades) / n,
        "net": sum(t["net"] for t in trades),
    }


def report_split(trades, split_date):
    """
    Split trades by split_date and report each half separately.

    Never returns or prints a single blended number without also
    showing the two halves individually.

    Args:
        trades: list of outcome records.
        split_date: cutoff date passed to split_by_date.

    Returns:
        Tuple of two stats records, one per half, from report().
    """
    before, after = split_by_date(trades, split_date)
    return report(before, "before"), report(after, "after")

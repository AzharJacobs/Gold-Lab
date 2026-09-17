"""
Backtest runner for strats/supply_demand_mtf.py.

Fetches H1/M30/M15/M1 XAUUSDm bars, generates signals from the
supply/demand MTF strategy, simulates outcomes on M1 bars (not M15 --
the strategy's stops are 2-4 points, tight enough that M15 granularity
produced ambiguous same-bar stop/target touches too often to trust),
and reports train/validate split per core/validate.py.

Usage: python run_supply_demand.py [months_back]
"""

import sys
from datetime import datetime

import pandas as pd

from core.simulate import simulate_all
from core.validate import split_by_date, report
from data import fetch
from strats.supply_demand_mtf import generate_signals

SYMBOL = "XAUUSDm"

# Measured live bid/ask spread on this demo account at time of writing
# was 0.26 (price units). Real trading costs (spread + any commission +
# slippage on fast moves) will typically run higher than a single
# snapshot. This is a documented estimate, not a guarantee -- see the
# written report for why it matters given how tight these stops are.
ROUND_TRIP_COST_XAU = 0.40


def _load_bars(symbol: str, timeframe: str, start: datetime, end: datetime):
    try:
        bars = fetch.load_from_cache(symbol, timeframe)
        if bars["time"].min() <= start and bars["time"].max() >= end:
            return bars
    except FileNotFoundError:
        pass
    bars = fetch.fetch(symbol, timeframe, start, end)
    fetch.save_to_cache(symbol, timeframe, bars)
    return bars


def _print_report(stats) -> None:
    print(
        f"{stats['label']:<22} "
        f"trades={stats['trades']:<6} "
        f"trades/mo={stats['trades_per_month']:<8.2f} "
        f"win%={stats['win_pct']:<7.1f} "
        f"avgR={stats['avg_r']:<8.3f} "
        f"net={stats['net']:<10.2f}"
    )


def run(months_back: int = 9, symbol: str = SYMBOL):
    end = datetime.now()
    start = end - pd.DateOffset(months=months_back)
    start = start.to_pydatetime()

    print(f"Fetching {symbol} H1/M30/M15/M1 from {start.date()} to {end.date()} ...")
    h1 = _load_bars(symbol, "H1", start, end)
    m30 = _load_bars(symbol, "M30", start, end)
    m15 = _load_bars(symbol, "M15", start, end)
    m1 = _load_bars(symbol, "M1", start, end)
    print(f"H1={len(h1)} M30={len(m30)} M15={len(m15)} M1={len(m1)} bars")

    signals = generate_signals({"H1": h1, "M30": m30, "M15": m15})
    print(f"signals generated: {len(signals)}")

    if not signals:
        print("No signals generated -- nothing to simulate.")
        return

    trades = simulate_all(signals, m1, round_trip_cost=ROUND_TRIP_COST_XAU)

    data_start = m1["time"].min()
    data_end = m1["time"].max()
    split_date = data_start + (data_end - data_start) / 2
    last6wk_start = data_end - pd.DateOffset(weeks=6)

    train, rest = split_by_date(trades, split_date)
    _, last6wk = split_by_date(trades, last6wk_start)

    print(f"\nstrategy=supply_demand_mtf symbol={symbol} "
          f"data={data_start.date()}..{data_end.date()} trades_total={len(trades)} "
          f"round_trip_cost={ROUND_TRIP_COST_XAU}")
    _print_report(report(train, "first half", data_start, split_date))
    _print_report(report(rest, "second half", split_date, data_end))
    _print_report(report(last6wk, "last 6 weeks", last6wk_start, data_end))

    stopped_then_would_have_hit_target = 0
    for sig, t in zip(signals, trades):
        if t["result"] != "stop":
            continue
        remaining = m1[m1["time"] > t["exit_time"]]
        remaining = remaining[remaining["time"] <= t["exit_time"] + pd.Timedelta(hours=48)]
        if remaining.empty:
            continue
        if sig["direction"] == "long":
            if (remaining["high"] >= sig["target"]).any():
                stopped_then_would_have_hit_target += 1
        else:
            if (remaining["low"] <= sig["target"]).any():
                stopped_then_would_have_hit_target += 1

    n_stopped = sum(1 for t in trades if t["result"] == "stop")
    print(f"\nof {n_stopped} stopped-out trades, {stopped_then_would_have_hit_target} "
          f"would have gone on to hit original target within 48h of the stop")

    out_path = "results/supply_demand_mtf_trades.csv"
    pd.DataFrame(trades).to_csv(out_path, index=False)
    print(f"\nfull trade log written to {out_path}")


if __name__ == "__main__":
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    run(mb)

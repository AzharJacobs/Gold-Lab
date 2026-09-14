"""
Entry point: load a strategy from strats/, run it, and print results
split into train / validate / last 6 months.

Usage: python run.py <strategy_name> [symbol]
"""

import importlib
import sys
from datetime import datetime

import pandas as pd

from core.simulate import simulate_all
from core.validate import split_by_date, report
from data import fetch

SYMBOL = "XAUUSDm"
TRAIN_VALIDATE_SPLIT = datetime(2025, 1, 1)
LAST_N_MONTHS = 6


def load_strategy(name: str):
    """
    Import strats/{name}.py and return its signal-generating function.

    A strategy module exports exactly one function: takes bars, returns
    a list of signals. Nothing else.
    """
    module = importlib.import_module(f"strats.{name}")
    return module.generate_signals


def _load_bars(symbol: str, timeframe: str, start: datetime, end: datetime):
    try:
        return fetch.load_from_cache(symbol, timeframe)
    except FileNotFoundError:
        bars = fetch.fetch(symbol, timeframe, start, end)
        fetch.save_to_cache(symbol, timeframe, bars)
        return bars


def _print_report(stats) -> None:
    print(
        f"{stats['label']:<20} "
        f"trades={stats['trades']:<6} "
        f"trades/mo={stats['trades_per_month']:<8.2f} "
        f"win%={stats['win_pct']:<7.1f} "
        f"avgR={stats['avg_r']:<8.3f} "
        f"net={stats['net']:<10.2f}"
    )


def run(strategy_name: str, symbol: str = SYMBOL):
    """
    Load bars, run the named strategy to get signals, simulate them,
    and print the train / validate / last-6-months report.

    Args:
        strategy_name: name of a module in strats/ (without .py).
        symbol: instrument to run against.
    """
    generate_signals = load_strategy(strategy_name)

    start = datetime(2018, 1, 1)
    end = datetime.now()
    h4 = _load_bars(symbol, "H4", start, end)
    m15 = _load_bars(symbol, "M15", start, end)

    signals = generate_signals({"H4": h4, "M15": m15})
    trades = simulate_all(signals, m15)

    if not trades:
        print(f"{strategy_name}: no trades generated.")
        return

    data_start = m15["time"].min()
    data_end = m15["time"].max()
    last6_start = data_end - pd.DateOffset(months=LAST_N_MONTHS)

    train, rest = split_by_date(trades, TRAIN_VALIDATE_SPLIT)
    _, last6 = split_by_date(trades, last6_start)

    print(f"strategy={strategy_name} symbol={symbol} "
          f"data={data_start.date()}..{data_end.date()} trades_total={len(trades)}")
    _print_report(report(train, "train (pre-2025)", data_start, TRAIN_VALIDATE_SPLIT))
    _print_report(report(rest, "validate (2025+)", TRAIN_VALIDATE_SPLIT, data_end))
    _print_report(report(last6, "last 6 months", last6_start, data_end))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <strategy_name> [symbol]")
        sys.exit(1)
    strategy = sys.argv[1]
    sym = sys.argv[2] if len(sys.argv) > 2 else SYMBOL
    run(strategy, sym)

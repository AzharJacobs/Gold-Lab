"""
Pull M5/M15/H1/H4/D1 bars from MT5 and cache them to data/cache/*.pkl.

Rules:
- Closed bars only. Never include the currently-forming bar.
- One cache file per (symbol, timeframe), e.g. data/cache/XAUUSD_H1.pkl.

MT5 login is read from environment variables (never hardcoded):
    MT5_DEMO_LOGIN, MT5_DEMO_PASSWORD, MT5_DEMO_SERVER
If unset, fetch() uses whatever account the local terminal is already
connected to.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

CACHE_DIR = Path(__file__).parent / "cache"

_TIMEFRAME_DELTA = {
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
    "D1": timedelta(days=1),
}

# Per-call history requests are capped by the terminal based on the
# requested span (independent of how much data actually exists in it) --
# a 365-day M5 request alone exceeds it. Long ranges are fetched in
# smaller chunks and concatenated; 90 days stays well under the cap even
# for the densest timeframe (M5).
_CHUNK = timedelta(days=90)


def _mt5_timeframe(mt5, timeframe: str):
    return {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }[timeframe]


def _connect(mt5):
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    login = os.environ.get("MT5_DEMO_LOGIN")
    password = os.environ.get("MT5_DEMO_PASSWORD")
    server = os.environ.get("MT5_DEMO_SERVER")
    if login and password and server:
        if not mt5.login(int(login), password=password, server=server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")


def fetch(symbol: str, timeframe: str, start, end):
    """
    Fetch closed bars for symbol/timeframe between start and end from MT5.

    Args:
        symbol: instrument symbol, e.g. "XAUUSD".
        timeframe: one of TIMEFRAMES.
        start: start datetime.
        end: end datetime.

    Returns:
        DataFrame of OHLCV bars, closed bars only, sorted by time ascending.
    """
    import MetaTrader5 as mt5

    _connect(mt5)
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MT5 symbol_select failed for {symbol}: {mt5.last_error()}")

        mt5_tf = _mt5_timeframe(mt5, timeframe)
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + _CHUNK, end)
            rates = mt5.copy_rates_range(symbol, mt5_tf, chunk_start, chunk_end)
            if rates is not None and len(rates):
                chunks.append(pd.DataFrame(rates))
            chunk_start = chunk_end
    finally:
        mt5.shutdown()

    if not chunks:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])

    bars = pd.concat(chunks, ignore_index=True)
    bars["time"] = pd.to_datetime(bars["time"], unit="s")
    bars = bars.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    # Closed bars only: drop any bar whose close time hasn't passed yet.
    bar_close = bars["time"] + _TIMEFRAME_DELTA[timeframe]
    bars = bars[bar_close <= datetime.now()].reset_index(drop=True)

    return bars[["time", "open", "high", "low", "close", "tick_volume"]]


def save_to_cache(symbol: str, timeframe: str, bars) -> None:
    """
    Save fetched bars to data/cache/{symbol}_{timeframe}.pkl.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_pickle(CACHE_DIR / f"{symbol}_{timeframe}.pkl")


def load_from_cache(symbol: str, timeframe: str):
    """
    Load cached bars for symbol/timeframe from data/cache/*.pkl.
    """
    return pd.read_pickle(CACHE_DIR / f"{symbol}_{timeframe}.pkl")

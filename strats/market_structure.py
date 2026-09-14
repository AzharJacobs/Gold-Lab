"""
Playbook A: 4H market structure + 15M entry confirmation.

Structure (4H):
    Swing highs/lows are fractals, 5 bars each side, confirmed only once
    those 5 confirming bars have closed.
    Bullish BOS = a 4H candle closes above the prior confirmed swing high.
    Bearish BOS = a 4H candle closes below the prior confirmed swing low.
    Wicks don't count -- only the close.
    After a bullish BOS: strong swing low = the lowest low between the
    prior swing low and the BOS candle. Weak swing high = the next
    confirmed swing high that forms after the BOS (the peak reached
    before the retracement). Bearish is the mirror.

Zone (4H):
    After a bullish BOS, the demand zone is the body range (open/close,
    not the wick) of the last down candle before the impulse leg that
    produced the BOS. After a bearish BOS, the supply zone is the body
    range of the last up candle before the impulse.

Entry (15M):
    Wait for price to retrace into the 4H zone, then require a 15M BOS
    in the 4H direction -- a 15M candle closing beyond the most recent
    confirmed 15M swing point. Enter at that candle's close.
    Stop: beyond the 15M strong swing point that formed inside the zone
    (the extreme 15M low/high reached while price was inside the zone).
    Target: the 4H weak swing high (bullish) or weak swing low (bearish).
    Timeout: exit after 96 M15 bars if neither stop nor target is hit.
    One trade per zone. A zone dies (no longer tradeable) the moment
    price closes through its strong swing point without having triggered
    an entry.

No look-ahead: a swing only exists once its 5 confirming bars have
closed, a zone only exists once its 4H BOS candle has closed, and a
zone's target is only usable once the weak swing that defines it has
itself confirmed.

This module exports one function, generate_signals(bars), which takes
bars and returns a list of signals. Nothing else.
"""

import bisect

FRACTAL_WING = 5
ENTRY_TIMEOUT_M15_BARS = 96


def _next_fractal_after(fractals, idx):
    """Return the first fractal dict with center > idx, or None."""
    centers = [f["center"] for f in fractals]
    pos = bisect.bisect_right(centers, idx)
    return fractals[pos] if pos < len(fractals) else None


def _last_body_before(bars, direction, before_idx):
    """
    Scan backward from before_idx (inclusive) for the nearest candle whose
    body is opposite `direction` (a down candle for "long", an up candle
    for "short") -- the last opposite candle before an impulsive move.

    Returns (zone_low, zone_high) from that candle's open/close, or None.
    """
    open_ = bars["open"].to_numpy()
    close = bars["close"].to_numpy()
    for k in range(before_idx, -1, -1):
        is_down = close[k] < open_[k]
        is_up = close[k] > open_[k]
        if (direction == "long" and is_down) or (direction == "short" and is_up):
            return min(open_[k], close[k]), max(open_[k], close[k])
    return None


def _find_fractals(bars):
    """
    Locate confirmed fractal swing highs/lows in a 4H or 15M bar series.

    A bar at index c is a swing high if its high is strictly greater than
    the highs of the FRACTAL_WING bars on each side of it; a swing low is
    the mirror on lows. It is only confirmed once those trailing
    FRACTAL_WING bars have closed, i.e. at index c + FRACTAL_WING.

    Args:
        bars: OHLC DataFrame sorted ascending by time.

    Returns:
        (highs, lows): each a list of dicts {center, price, confirm},
        sorted ascending by center index. "confirm" is the bar index at
        which the swing becomes usable (no look-ahead before then).
    """
    wing = FRACTAL_WING
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    n = len(bars)

    highs, lows = [], []
    for c in range(wing, n - wing):
        if high[c] > high[c - wing:c].max() and high[c] > high[c + 1:c + wing + 1].max():
            highs.append({"center": c, "price": high[c], "confirm": c + wing})
        if low[c] < low[c - wing:c].min() and low[c] < low[c + 1:c + wing + 1].min():
            lows.append({"center": c, "price": low[c], "confirm": c + wing})

    return highs, lows


def _detect_bos_events(bars, highs, lows):
    """
    Walk a 4H bar series and detect break-of-structure events.

    At each bar, the most recently confirmed swing high/low becomes the
    active reference. A bullish BOS fires the first time a candle closes
    above the active reference swing high; a bearish BOS fires the first
    time a candle closes below the active reference swing low. Firing
    clears that side's reference so the same level cannot re-fire, and a
    new reference must confirm before the next BOS on that side.

    Args:
        bars: 4H OHLC DataFrame sorted ascending by time.
        highs, lows: fractal lists from _find_fractals(bars).

    Returns:
        List of event dicts in chronological order:
        {type: "bullish"|"bearish", bos_idx, prior_low, prior_high}
        where prior_low/prior_high are the fractal dicts active (possibly
        None) on the non-broken side at the moment of the event.
    """
    close = bars["close"].to_numpy()
    n = len(bars)

    events = []
    hi_ptr = lo_ptr = 0
    active_high = active_low = None

    for i in range(n):
        while hi_ptr < len(highs) and highs[hi_ptr]["confirm"] <= i:
            active_high = highs[hi_ptr]
            hi_ptr += 1
        while lo_ptr < len(lows) and lows[lo_ptr]["confirm"] <= i:
            active_low = lows[lo_ptr]
            lo_ptr += 1

        if active_high is not None and close[i] > active_high["price"]:
            events.append({
                "type": "bullish",
                "bos_idx": i,
                "prior_low": active_low,
                "prior_high": active_high,
            })
            active_high = None
        elif active_low is not None and close[i] < active_low["price"]:
            events.append({
                "type": "bearish",
                "bos_idx": i,
                "prior_low": active_low,
                "prior_high": active_high,
            })
            active_low = None

    return events


def _build_zone(event, bars, highs, lows):
    """
    Turn a BOS event into a tradeable 4H zone, per the Zone/Structure rules.

    Args:
        event: one event dict from _detect_bos_events.
        bars: the same 4H OHLC DataFrame.
        highs, lows: fractal lists from _find_fractals(bars).

    Returns:
        Zone dict: {direction: "long"|"short", bos_idx, bos_time,
        strong_price, zone_low, zone_high, target_price,
        target_ready_time}, or None if a required piece (prior swing,
        weak swing, or a qualifying body candle) doesn't exist.
    """
    low = bars["low"].to_numpy()
    high = bars["high"].to_numpy()
    time = bars["time"]
    bos_idx = event["bos_idx"]

    if event["type"] == "bullish":
        prior_low = event["prior_low"]
        if prior_low is None:
            return None

        window_start = prior_low["center"]
        window = low[window_start:bos_idx + 1]
        strong_idx = window_start + int(window.argmin())
        strong_price = float(window.min())

        weak_swing = _next_fractal_after(highs, bos_idx)
        if weak_swing is None:
            return None

        body = _last_body_before(bars, "long", strong_idx)
        if body is None:
            return None
        zone_low, zone_high = body

        return {
            "direction": "long",
            "bos_idx": bos_idx,
            "bos_time": time.iloc[bos_idx],
            "strong_price": strong_price,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "target_price": weak_swing["price"],
            "target_ready_time": time.iloc[weak_swing["confirm"]],
        }

    else:
        prior_high = event["prior_high"]
        if prior_high is None:
            return None

        window_start = prior_high["center"]
        window = high[window_start:bos_idx + 1]
        strong_idx = window_start + int(window.argmax())
        strong_price = float(window.max())

        weak_swing = _next_fractal_after(lows, bos_idx)
        if weak_swing is None:
            return None

        body = _last_body_before(bars, "short", strong_idx)
        if body is None:
            return None
        zone_low, zone_high = body

        return {
            "direction": "short",
            "bos_idx": bos_idx,
            "bos_time": time.iloc[bos_idx],
            "strong_price": strong_price,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "target_price": weak_swing["price"],
            "target_ready_time": time.iloc[weak_swing["confirm"]],
        }


def _monitor_zone(zone, m15, m15_highs, m15_lows):
    """
    Watch 15M bars after a zone forms for the entry trigger, per Entry.

    Steps, in order, bar by bar starting after the zone's 4H BOS candle
    closes: check invalidation (a 15M close through the zone's
    strong_price kills the zone), check for the first retracement touch
    into the zone, then -- once touched -- check for a 15M BOS beyond the
    most recent confirmed 15M swing point in the zone's direction, with
    the target already confirmed. The first qualifying bar is the entry.

    Args:
        zone: a zone dict from _build_zone.
        m15: 15M OHLC DataFrame sorted ascending by time.
        m15_highs, m15_lows: fractal lists from _find_fractals(m15).

    Returns:
        A signal dict {time, direction, entry, stop, target,
        timeout_bars}, or None if the zone is invalidated or never
        triggers before the data ends.
    """
    time = m15["time"]
    low = m15["low"].to_numpy()
    high = m15["high"].to_numpy()
    close = m15["close"].to_numpy()
    n = len(m15)

    start_pos = int(time.searchsorted(zone["bos_time"], side="right"))

    touched = False
    touch_idx = None
    hi_ptr = lo_ptr = 0
    active_high = active_low = None

    for i in range(start_pos, n):
        while hi_ptr < len(m15_highs) and m15_highs[hi_ptr]["confirm"] <= i:
            active_high = m15_highs[hi_ptr]
            hi_ptr += 1
        while lo_ptr < len(m15_lows) and m15_lows[lo_ptr]["confirm"] <= i:
            active_low = m15_lows[lo_ptr]
            lo_ptr += 1

        if zone["direction"] == "long" and close[i] < zone["strong_price"]:
            return None
        if zone["direction"] == "short" and close[i] > zone["strong_price"]:
            return None

        if not touched:
            overlap = low[i] <= zone["zone_high"] and high[i] >= zone["zone_low"]
            if overlap:
                touched = True
                touch_idx = i
            continue

        if zone["direction"] == "long":
            if (
                active_high is not None
                and close[i] > active_high["price"]
                and time.iloc[i] >= zone["target_ready_time"]
            ):
                stop = float(low[touch_idx:i + 1].min())
                return {
                    "time": time.iloc[i],
                    "direction": "long",
                    "entry": float(close[i]),
                    "stop": stop,
                    "target": zone["target_price"],
                    "timeout_bars": ENTRY_TIMEOUT_M15_BARS,
                }
        else:
            if (
                active_low is not None
                and close[i] < active_low["price"]
                and time.iloc[i] >= zone["target_ready_time"]
            ):
                stop = float(high[touch_idx:i + 1].max())
                return {
                    "time": time.iloc[i],
                    "direction": "short",
                    "entry": float(close[i]),
                    "stop": stop,
                    "target": zone["target_price"],
                    "timeout_bars": ENTRY_TIMEOUT_M15_BARS,
                }

    return None


def generate_signals(bars):
    """
    Playbook A entry point. Takes bars, returns a list of signals.

    Args:
        bars: dict with keys "H4" and "M15", each an OHLC DataFrame
            (columns: time, open, high, low, close) sorted ascending by
            time, closed bars only.

    Returns:
        List of signal dicts {time, direction, entry, stop, target,
        timeout_bars}, sorted ascending by time. One signal per zone at
        most.
    """
    h4 = bars["H4"].reset_index(drop=True)
    m15 = bars["M15"].reset_index(drop=True)

    h4_highs, h4_lows = _find_fractals(h4)
    m15_highs, m15_lows = _find_fractals(m15)

    events = _detect_bos_events(h4, h4_highs, h4_lows)

    signals = []
    for event in events:
        zone = _build_zone(event, h4, h4_highs, h4_lows)
        if zone is None:
            continue
        signal = _monitor_zone(zone, m15, m15_highs, m15_lows)
        if signal is not None:
            signals.append(signal)

    signals.sort(key=lambda s: s["time"])
    return signals

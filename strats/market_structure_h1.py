"""
Playbook A (H1/M5 variant): H1 market structure + M5 entry confirmation.

Structure (H1):
    Swing highs/lows are fractals, 5 bars each side, confirmed only once
    those 5 confirming bars have closed.
    Bullish BOS = an H1 candle closes above the prior confirmed swing high.
    Bearish BOS = an H1 candle closes below the prior confirmed swing low.
    Wicks don't count -- only the close.
    After a bullish BOS: strong swing low = the lowest low between the
    prior swing low and the BOS candle. Weak swing high = the next
    confirmed swing high that forms after the BOS (the peak reached
    before the retracement). Bearish is the mirror.

Zone (H1):
    After a BOS, the impulse leg runs from the strong swing point to the
    BOS candle. Walking forward from the strong swing point, the impulse
    is judged to begin at the first candle whose body (abs(close-open))
    is at least 1.5x the mean body size of the 20 candles ending at the
    strong swing point. The zone is the full high-to-low range (wicks
    included) of every candle from the strong swing point up to but not
    including that impulse candle -- the last consolidation before the
    move. If no candle up to the BOS candle meets the threshold, the
    zone is the range of the strong swing point through the next 2
    candles.

Entry (M5):
    Wait for price to retrace into the H1 zone, then require an M5 BOS
    in the H1 direction -- an M5 candle closing beyond the most recent
    confirmed M5 swing point. Enter at that candle's close.
    Stop: beyond the M5 strong swing point that formed inside the zone
    (the extreme M5 low/high reached while price was inside the zone).
    Target: the H1 weak swing high (bullish) or weak swing low (bearish).
    Timeout: exit after 288 M5 bars if neither stop nor target is hit
    (the same 24-hour wall-clock duration as 96 M15 bars).
    One trade per zone. A zone dies (no longer tradeable) the moment
    price closes through its strong swing point without having triggered
    an entry.

No look-ahead: a swing only exists once its 5 confirming bars have
closed, a zone only exists once its H1 BOS candle has closed, and a
zone's target is only usable once the weak swing that defines it has
itself confirmed.

This module exports one function, generate_signals(bars), which takes
bars and returns a list of signals. Nothing else.
"""

import bisect

FRACTAL_WING = 5
ENTRY_TIMEOUT_M5_BARS = 288


def _next_fractal_after(fractals, idx):
    """Return the first fractal dict with center > idx, or None."""
    centers = [f["center"] for f in fractals]
    pos = bisect.bisect_right(centers, idx)
    return fractals[pos] if pos < len(fractals) else None


def _consolidation_range_before_impulse(bars, strong_idx, bos_idx):
    """
    Find the last consolidation range before the impulse leg running from
    strong_idx to bos_idx.

    Walking forward from strong_idx, the impulse leg is judged to begin
    at the first candle whose body (abs(close-open)) is at least 1.5x
    the mean body size of the 20 candles ending at strong_idx. The zone
    is the full high-to-low range (wicks included) of every candle from
    strong_idx up to but not including that impulse candle. If no
    candle up to and including bos_idx meets the threshold, the zone is
    the range of strong_idx through strong_idx + 2. The span never
    includes any candle with index greater than bos_idx -- a zone must
    be fully knowable at bos_time.

    Returns (zone_low, zone_high).
    """
    open_ = bars["open"].to_numpy()
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()

    baseline_start = max(0, strong_idx - 19)
    baseline_bodies = abs(close[baseline_start:strong_idx + 1] - open_[baseline_start:strong_idx + 1])
    threshold = 1.5 * baseline_bodies.mean()

    impulse_idx = None
    for idx in range(strong_idx + 1, bos_idx + 1):
        if abs(close[idx] - open_[idx]) >= threshold:
            impulse_idx = idx
            break

    if impulse_idx is None:
        span_end = min(strong_idx + 2, bos_idx)
        span_indices = range(strong_idx, span_end + 1)
    else:
        span_indices = range(strong_idx, impulse_idx)

    span_low = min(low[i] for i in span_indices)
    span_high = max(high[i] for i in span_indices)
    return span_low, span_high


def _find_fractals(bars):
    """
    Locate confirmed fractal swing highs/lows in an H1 or M5 bar series.

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
    Walk an H1 bar series and detect break-of-structure events.

    At each bar, the most recently confirmed swing high/low becomes the
    active reference. A bullish BOS fires the first time a candle closes
    above the active reference swing high; a bearish BOS fires the first
    time a candle closes below the active reference swing low. Firing
    clears that side's reference so the same level cannot re-fire, and a
    new reference must confirm before the next BOS on that side.

    Args:
        bars: H1 OHLC DataFrame sorted ascending by time.
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
    Turn a BOS event into a tradeable H1 zone, per the Zone/Structure rules.

    Args:
        event: one event dict from _detect_bos_events.
        bars: the same H1 OHLC DataFrame.
        highs, lows: fractal lists from _find_fractals(bars).

    Returns:
        Zone dict: {direction: "long"|"short", bos_idx, bos_time,
        strong_price, zone_low, zone_high, target_price,
        target_ready_time}, or None if a required piece (prior swing or
        weak swing) doesn't exist.
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

        zone_low, zone_high = _consolidation_range_before_impulse(bars, strong_idx, bos_idx)

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

        zone_low, zone_high = _consolidation_range_before_impulse(bars, strong_idx, bos_idx)

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


def _monitor_zone(zone, m5, m5_highs, m5_lows):
    """
    Watch M5 bars after a zone forms for the entry trigger, per Entry.

    Steps, in order, bar by bar starting after the zone's H1 BOS candle
    closes: check invalidation (an M5 close through the zone's
    strong_price kills the zone), check for the first retracement touch
    into the zone, then -- once touched -- check for an M5 BOS beyond the
    most recent confirmed M5 swing point in the zone's direction, with
    the target already confirmed. The first qualifying bar is the entry.

    Args:
        zone: a zone dict from _build_zone.
        m5: M5 OHLC DataFrame sorted ascending by time.
        m5_highs, m5_lows: fractal lists from _find_fractals(m5).

    Returns:
        A signal dict {time, direction, entry, stop, target,
        timeout_bars}, or None if the zone is invalidated or never
        triggers before the data ends.
    """
    time = m5["time"]
    low = m5["low"].to_numpy()
    high = m5["high"].to_numpy()
    close = m5["close"].to_numpy()
    n = len(m5)

    start_pos = int(time.searchsorted(zone["bos_time"], side="right"))

    touched = False
    touch_idx = None
    hi_ptr = lo_ptr = 0
    active_high = active_low = None

    for i in range(start_pos, n):
        while hi_ptr < len(m5_highs) and m5_highs[hi_ptr]["confirm"] <= i:
            active_high = m5_highs[hi_ptr]
            hi_ptr += 1
        while lo_ptr < len(m5_lows) and m5_lows[lo_ptr]["confirm"] <= i:
            active_low = m5_lows[lo_ptr]
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
                    "timeout_bars": ENTRY_TIMEOUT_M5_BARS,
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
                    "timeout_bars": ENTRY_TIMEOUT_M5_BARS,
                }

    return None


def _generate_candidates(bars):
    """
    Internal: run structure/zone/entry detection and return every
    triggered signal before the invalid-target and one-per-bar
    corrections applied in generate_signals, paired with the bos_idx of
    the zone that produced it (used to break same-bar ties).
    """
    h1 = bars["H1"].reset_index(drop=True)
    m5 = bars["M5"].reset_index(drop=True)

    h1_highs, h1_lows = _find_fractals(h1)
    m5_highs, m5_lows = _find_fractals(m5)

    events = _detect_bos_events(h1, h1_highs, h1_lows)

    candidates = []
    for event in events:
        zone = _build_zone(event, h1, h1_highs, h1_lows)
        if zone is None:
            continue
        signal = _monitor_zone(zone, m5, m5_highs, m5_lows)
        if signal is not None:
            candidates.append((signal, zone["bos_idx"]))

    candidates.sort(key=lambda pair: pair[0]["time"])
    return candidates


def _has_valid_target(signal):
    """
    A weak swing high/low is a liquidity target in the trade direction --
    it must sit beyond entry on the correct side. If it doesn't (an
    immediate, violent retracement can leave the next confirmed fractal
    below entry on a buy, or above it on a sell), the setup is invalid.
    """
    if signal["direction"] == "long":
        return signal["target"] > signal["entry"]
    return signal["target"] < signal["entry"]


def _drop_same_bar_duplicates(candidates):
    """
    Only one position is held at a time: if multiple pending zones
    trigger their entry on the same M5 bar, keep the signal from the
    zone whose H1 BOS was most recent and drop the rest.
    """
    by_bar = {}
    for signal, bos_idx in candidates:
        key = signal["time"]
        if key not in by_bar or bos_idx > by_bar[key][1]:
            by_bar[key] = (signal, bos_idx)
    return list(by_bar.values())


def generate_signals(bars):
    """
    Playbook A (H1/M5 variant) entry point. Takes bars, returns a list of
    signals.

    Two corrections are applied to the raw triggered signals: a signal
    whose target is on the wrong side of entry is dropped (not moved,
    not substituted), and when more than one signal fires on the same
    M5 bar, only the one from the most recently broken H1 structure is
    kept.

    Args:
        bars: dict with keys "H1" and "M5", each an OHLC DataFrame
            (columns: time, open, high, low, close) sorted ascending by
            time, closed bars only.

    Returns:
        List of signal dicts {time, direction, entry, stop, target,
        timeout_bars}, sorted ascending by time. At most one signal per
        zone, and at most one signal per M5 bar.
    """
    candidates = _generate_candidates(bars)
    candidates = [(s, b) for s, b in candidates if _has_valid_target(s)]
    candidates = _drop_same_bar_duplicates(candidates)

    signals = [s for s, _ in candidates]
    signals.sort(key=lambda s: s["time"])
    return signals

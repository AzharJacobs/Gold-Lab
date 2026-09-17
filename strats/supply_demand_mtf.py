"""
Supply/demand multi-timeframe fade strategy (H1 / M30 / M15).

This is a mechanical PROXY of a discretionary strategy, built from the
trader's stated rules plus a set of explicit assumptions where those
rules were ambiguous or contradictory. Every assumption is marked
ASSUMPTION below. A backtest of this file tests the proxy, not the
trader's actual discretionary chart-reading -- results should be read
with that gap in mind.

Two entry models are supported (entry_trigger param), because the
trader's own rules conflict about how a zone touch becomes a trade --
see the "ENTRY MODELS" section below. TUNING DISCLOSURE: the single-
candle zone width and the 3.0-point default stop buffer were both
chosen specifically to reproduce the trader's stated 2-4 point stops
in their three real examples. That is fitting to their description,
not an independently-derived rule. Treat any result built on the
default stop_buffer=3.0 with that in mind.

Zone (SBR/RBS), per TF (H1, M30, M15), independently:
    Swing highs/lows are fractals, 5 bars each side, confirmed once
    those 5 confirming bars have closed. A bullish BOS = a candle
    closes above the prior confirmed swing high; bearish BOS = a
    candle closes below the prior confirmed swing low. Wicks don't
    count for BOS, only the close.
    strong_price = the most extreme wick between the prior opposite
    swing and the BOS candle (the origin of the impulse leg).
    The zone itself is that single origin candle's own high/low wick
    range -- not a multi-candle consolidation (ASSUMPTION -- see
    TUNING DISCLOSURE above).

No look-ahead, corrected: a BOS is only known once its own candle
CLOSES, i.e. at bos_time + one bar duration on that timeframe -- not
at bos_time itself, which is the bar's OPEN. Both the bias filter and
target-zone availability now gate on usable_time = bos_time +
duration. (Previously both compared against bos_time directly, which
let a zone influence trades up to one full bar early -- up to 59
minutes on H1. This was a real bug, not a stylistic choice.)

Bias, H1 and M30 only (ASSUMPTION -- M15 never gates bias, only 1h/30m
per the trader's rule):
    bias(TF, t) = direction of the most recently formed BOS zone on
    that TF with usable_time <= t. A trade's direction must equal
    both H1 bias and M30 bias, evaluated at the exact moment of entry,
    with zero bars of tolerance (ASSUMPTION: the trader's "bias can
    resolve retroactively" rule was left without a number for how
    much lag is allowed, so lag is fixed at 0).

Zone structure, independent of entry model:
    A zone is "touched" when price first wicks into [zone_low,
    zone_high] after forming (each touch is one contiguous excursion:
    price enters, then later leaves, before the next touch can start).
    It is invalidated as of the touch that is its 3rd distinct touch
    (dies AT the 3rd touch -- that touch is not itself tradeable), or
    immediately if any candle CLOSES beyond strong_price. Only touches
    1 and 2 are ever tradeable. This lifecycle (touches, death_time)
    is computed once, the same way regardless of which entry model is
    used -- a zone's availability as a TARGET for some other trade
    should not depend on which entry model happens to be under test.

ENTRY MODELS (entry_trigger param -- this is the fix for the
survivorship-bias bug: the original version only ever created a
signal when a touch went on to produce a confirming candle, which
silently drops every touch that just blew through the zone and lost.
That's the favorable entry price of a touch model combined with the
selectivity of a confirmation model. Pick one, not a blend of both):

  "plain_touch" (Model A): every tradeable touch (1st and 2nd) is a
    trade. No pattern filter. Entry = the zone edge nearest the
    approach direction, at the moment of that touch (limit order
    sitts at the edge, fills on contact). Stop = that touch bar's own
    wick, +/- stop_buffer, clipped inside the zone. Expected to look
    much worse than the confirmation models -- that's deliberately
    the point of including it.

  "engulf" (Model B): on touches 1 or 2, scan forward bar-by-bar
    while price stays inside the zone; the first full-body engulfing
    candle in the trade direction (open <= prior close and close >=
    prior open, mirrored for bearish) whose own close is back outside
    the zone confirms. If price leaves the zone without one, that
    touch produces no trade (real difference from "plain_touch": a
    losing touch that never confirms is genuinely a no-trade under
    this model, not a hidden win). Entry = the confirming candle's own
    close. Stop = that same candle's own wick, +/- stop_buffer,
    clipped inside the zone. Only the first touch (of up to 2) that
    confirms produces a trade -- one trade per zone.

  "pin" (Model B variant): same scan, but the pattern is a pin bar --
    a wick on the reversal side at least 2x the candle's body, body
    non-dominant, closing back outside the zone.

Target modes (target_mode param):
  "nearest_zone": the near edge of the nearest still-valid opposing
    zone (any of H1/M30/M15) with usable_time <= entry_time and not
    yet dead at entry_time, beyond entry in the trade's favor
    (ASSUMPTION: "the opposite level" read literally as the nearest
    opposing zone). If none exists, the signal is dropped.
  "fixed": entry +/- target_fixed_pts, direction-aware. Needs
    target_fixed_pts set.

Entry timeout: ENTRY_TIMEOUT_M1_BARS (2880 = ~48 hours), because
signals from this module are simulated on M1 bars elsewhere in this
project -- the timeout must be in the SAME bar unit as the series
being simulated. (A previous version set this to 96 intending "about
a day" on the zone's own H1/M30/M15 timeframe, but fed it to an M1
simulation, where it meant 96 minutes -- a real bug that materially
inflated the "timeout" outcome bucket in earlier results.)

This module exports:
  generate_signals(bars, entry_trigger="engulf", stop_buffer=3.0,
                    target_mode="nearest_zone", target_fixed_pts=None)
    -- the full pipeline in one call, using the given parameters.
  prepare_zones(bars) / signals_from_zones(zone_base, ...)
    -- split apart for sweeps: prepare_zones does the (entry-model-
    independent, expensive) fractal/BOS/structure work once;
    signals_from_zones applies a given parameter combination to that
    precomputed structure. generate_signals is just
    signals_from_zones(prepare_zones(bars), ...).
"""

import bisect

import pandas as pd

FRACTAL_WING = 5
ZONE_TOUCH_LIMIT = 3
STOP_ZONE_MARGIN = 0.05
ENTRY_TIMEOUT_M1_BARS = 2880
BIAS_TFS = ("H1", "M30")
ZONE_TFS = ("H1", "M30", "M15")
TF_DURATION = {
    "H1": pd.Timedelta(hours=1),
    "M30": pd.Timedelta(minutes=30),
    "M15": pd.Timedelta(minutes=15),
}


def _find_fractals(bars):
    """Confirmed fractal swing highs/lows, 5 bars each side. See module docstring."""
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
    """Break-of-structure events, close-only. See module docstring."""
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
            events.append({"type": "bullish", "bos_idx": i, "prior_low": active_low, "prior_high": active_high})
            active_high = None
        elif active_low is not None and close[i] < active_low["price"]:
            events.append({"type": "bearish", "bos_idx": i, "prior_low": active_low, "prior_high": active_high})
            active_low = None

    return events


def _build_zones(tf, bars):
    """
    Build every SBR/RBS zone for one timeframe's bar series, in
    chronological order of bos_time.

    Returns list of zone dicts: {tf, direction, bos_idx, bos_time,
    usable_time, strong_price, zone_low, zone_high}. usable_time is
    bos_time + one bar duration on this TF -- the first moment the
    BOS is actually known (its candle has closed).
    """
    highs, lows = _find_fractals(bars)
    events = _detect_bos_events(bars, highs, lows)
    low = bars["low"].to_numpy()
    high = bars["high"].to_numpy()
    time = bars["time"]
    duration = TF_DURATION[tf]

    zones = []
    for event in events:
        bos_idx = event["bos_idx"]
        if event["type"] == "bullish":
            prior_low = event["prior_low"]
            if prior_low is None:
                continue
            window_start = prior_low["center"]
            window = low[window_start:bos_idx + 1]
            strong_idx = window_start + int(window.argmin())
            strong_price = float(window.min())
            direction = "long"
        else:
            prior_high = event["prior_high"]
            if prior_high is None:
                continue
            window_start = prior_high["center"]
            window = high[window_start:bos_idx + 1]
            strong_idx = window_start + int(window.argmax())
            strong_price = float(window.max())
            direction = "short"

        bos_time = time.iloc[bos_idx]
        zones.append({
            "tf": tf,
            "direction": direction,
            "bos_idx": bos_idx,
            "bos_time": bos_time,
            "usable_time": bos_time + duration,
            "strong_price": strong_price,
            "zone_low": float(low[strong_idx]),
            "zone_high": float(high[strong_idx]),
        })

    return zones


def _pattern_engulf(direction, open_, high, low, close, i):
    """Full-body engulfing candle at index i, in the given direction."""
    if i < 1:
        return False
    if direction == "long":
        return close[i] > open_[i] and open_[i] <= close[i - 1] and close[i] >= open_[i - 1]
    return close[i] < open_[i] and open_[i] >= close[i - 1] and close[i] <= open_[i - 1]


def _pattern_pin(direction, open_, high, low, close, i):
    """Pin bar: reversal-side wick >= 2x body, body not dominant, at index i."""
    body = abs(close[i] - open_[i])
    if direction == "long":
        wick = min(open_[i], close[i]) - low[i]
        other_wick = high[i] - max(open_[i], close[i])
    else:
        wick = high[i] - max(open_[i], close[i])
        other_wick = min(open_[i], close[i]) - low[i]
    return wick >= 2 * max(body, 1e-9) and wick > other_wick


_PATTERNS = {"engulf": _pattern_engulf, "pin": _pattern_pin}


def _zone_structure(zone, bars):
    """
    Walk a zone's own timeframe forward from its BOS candle and resolve
    the entry-model-INDEPENDENT parts of its lifecycle: every tradeable
    touch (1st and 2nd -- the 3rd kills the zone before it can be
    used), and death_time (close-through or 3rd touch, or None if the
    zone survives to the end of the data untouched).

    Returns the zone dict augmented with: touches (list of bar indices,
    each the first bar of a tradeable excursion into the zone) and
    death_time.
    """
    time = bars["time"]
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    n = len(bars)

    zone_low, zone_high = zone["zone_low"], zone["zone_high"]
    direction = zone["direction"]

    start_pos = int(time.searchsorted(zone["bos_time"], side="right"))

    result = dict(zone)
    result["touches"] = []
    result["death_time"] = None

    inside = False
    touch_count = 0

    for i in range(start_pos, n):
        if direction == "long" and close[i] < zone["strong_price"]:
            result["death_time"] = time.iloc[i]
            return result
        if direction == "short" and close[i] > zone["strong_price"]:
            result["death_time"] = time.iloc[i]
            return result

        overlap = low[i] <= zone_high and high[i] >= zone_low

        if not inside and overlap:
            inside = True
            touch_count += 1
            if touch_count >= ZONE_TOUCH_LIMIT:
                result["death_time"] = time.iloc[i]
                return result
            result["touches"].append(i)
        elif inside and not overlap:
            inside = False

    return result


def _scan_confirmation(pattern_fn, direction, zone_low, zone_high, open_, high, low, close, start_idx, n):
    """
    From start_idx (a touch bar), scan forward while price stays inside
    the zone for the first bar matching pattern_fn whose close is back
    outside the zone. Returns that bar's index, or None if the
    excursion ends (price leaves the zone) without confirming.
    """
    for i in range(start_idx, n):
        overlap = low[i] <= zone_high and high[i] >= zone_low
        if i > start_idx and not overlap:
            return None
        if pattern_fn(direction, open_, high, low, close, i):
            outside = close[i] > zone_high if direction == "long" else close[i] < zone_low
            if outside:
                return i
    return None


def _zone_candidate_signals(zone, bars, entry_trigger, stop_buffer):
    """
    Raw candidate signals for one zone (already carrying "touches" and
    "death_time" from _zone_structure), under the given entry model.
    Each candidate: {time, direction, entry, stop, tf}.
    """
    time = bars["time"]
    open_ = bars["open"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    n = len(bars)

    direction = zone["direction"]
    zone_low, zone_high = zone["zone_low"], zone["zone_high"]
    candidates = []

    if entry_trigger == "plain_touch":
        for touch_idx in zone["touches"]:
            entry = zone_high if direction == "long" else zone_low
            if direction == "long":
                stop = max(float(low[touch_idx]) - stop_buffer, zone_low + STOP_ZONE_MARGIN)
            else:
                stop = min(float(high[touch_idx]) + stop_buffer, zone_high - STOP_ZONE_MARGIN)
            candidates.append({
                "time": time.iloc[touch_idx], "direction": direction,
                "entry": float(entry), "stop": float(stop), "tf": zone["tf"],
            })
        return candidates

    pattern_fn = _PATTERNS[entry_trigger]
    for touch_idx in zone["touches"]:
        confirm_idx = _scan_confirmation(pattern_fn, direction, zone_low, zone_high, open_, high, low, close, touch_idx, n)
        if confirm_idx is None:
            continue
        entry = float(close[confirm_idx])
        if direction == "long":
            stop = max(float(low[confirm_idx]) - stop_buffer, zone_low + STOP_ZONE_MARGIN)
        else:
            stop = min(float(high[confirm_idx]) + stop_buffer, zone_high - STOP_ZONE_MARGIN)
        candidates.append({
            "time": time.iloc[confirm_idx], "direction": direction,
            "entry": entry, "stop": float(stop), "tf": zone["tf"],
        })
        break  # one trade per zone for confirmation models

    return candidates


def _bias_lookup(zones):
    """Step-function lookup keyed on usable_time (see module docstring)."""
    ordered = sorted(zones, key=lambda z: z["usable_time"])
    times = [z["usable_time"] for z in ordered]
    directions = [z["direction"] for z in ordered]
    return times, directions


def _bias_at(times, directions, t):
    pos = bisect.bisect_right(times, t) - 1
    if pos < 0:
        return None
    return directions[pos]


def _find_target(opposing_pool, direction, entry_time, entry_price, target_mode, target_fixed_pts):
    if target_mode == "fixed":
        return entry_price + target_fixed_pts if direction == "long" else entry_price - target_fixed_pts

    best = None
    best_dist = None
    for z in opposing_pool:
        if z["usable_time"] > entry_time:
            continue
        if z["death_time"] is not None and z["death_time"] <= entry_time:
            continue

        if direction == "long":
            edge = z["zone_low"]
            if edge <= entry_price:
                continue
            dist = edge - entry_price
        else:
            edge = z["zone_high"]
            if edge >= entry_price:
                continue
            dist = entry_price - edge

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = edge

    return best


def prepare_zones(bars):
    """
    Entry-model-independent groundwork: build zones per TF and resolve
    their touch/death structure. Expensive relative to
    signals_from_zones, so sweeps should call this once and reuse it.

    Args:
        bars: dict with keys "H1", "M30", "M15", each an OHLC
            DataFrame sorted ascending by time, closed bars only.

    Returns: dict with "frames" (the reset-index bar frames, needed by
    signals_from_zones for array access) and "zones" (dict tf ->
    list of zone dicts with touches/death_time attached).
    """
    frames = {tf: bars[tf].reset_index(drop=True) for tf in ZONE_TFS}
    raw_zones = {tf: _build_zones(tf, frames[tf]) for tf in ZONE_TFS}
    zones = {tf: [_zone_structure(z, frames[tf]) for z in raw_zones[tf]] for tf in ZONE_TFS}
    return {"frames": frames, "zones": zones}


def signals_from_zones(zone_base, entry_trigger="engulf", stop_buffer=3.0,
                        target_mode="nearest_zone", target_fixed_pts=None):
    """
    Apply one (entry_trigger, stop_buffer, target_mode) combination to
    already-prepared zone structure. See module docstring for the
    meaning of each entry_trigger / target_mode value.
    """
    if target_mode == "fixed" and target_fixed_pts is None:
        raise ValueError("target_mode='fixed' requires target_fixed_pts")

    frames = zone_base["frames"]
    zones = zone_base["zones"]

    bias_lookup = {tf: _bias_lookup(zones[tf]) for tf in BIAS_TFS}
    all_zones = [z for tf in ZONE_TFS for z in zones[tf]]

    candidates = []
    for tf in ZONE_TFS:
        for z in zones[tf]:
            for raw in _zone_candidate_signals(z, frames[tf], entry_trigger, stop_buffer):
                direction = raw["direction"]
                aligned = all(
                    _bias_at(*bias_lookup[btf], raw["time"]) == direction
                    for btf in BIAS_TFS
                )
                if not aligned:
                    continue

                opposing_pool = [oz for oz in all_zones if oz["direction"] != direction and oz is not z]
                target = _find_target(opposing_pool, direction, raw["time"], raw["entry"], target_mode, target_fixed_pts)
                if target is None:
                    continue

                signal = {
                    "time": raw["time"], "direction": direction,
                    "entry": raw["entry"], "stop": raw["stop"], "target": target,
                    "timeout_bars": ENTRY_TIMEOUT_M1_BARS,
                }
                candidates.append(signal)

    candidates.sort(key=lambda s: s["time"])

    by_bar = {}
    for s in candidates:
        key = s["time"]
        if key not in by_bar:
            by_bar[key] = s
    signals = list(by_bar.values())
    signals.sort(key=lambda s: s["time"])
    return signals


def generate_signals(bars, entry_trigger="engulf", stop_buffer=3.0,
                      target_mode="nearest_zone", target_fixed_pts=None):
    """
    Supply/demand multi-timeframe entry point. Takes bars, returns a
    list of signals. See module docstring for parameter meanings.
    """
    zone_base = prepare_zones(bars)
    return signals_from_zones(zone_base, entry_trigger, stop_buffer, target_mode, target_fixed_pts)

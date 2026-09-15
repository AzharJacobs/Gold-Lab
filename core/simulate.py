"""
Walk-forward simulation of a single signal against bars.

Rule: if both stop and target are touched on the same bar, the stop is
assumed to have been hit first (conservative fill).
"""

from core.costs import apply_cost


def simulate_trade(signal, bars, round_trip_cost=None):
    """
    Walk forward through bars from the signal's entry time and determine outcome.

    Args:
        signal: dict-like with keys {time, direction, entry, stop, target}.
            direction is "long" or "short". May optionally include
            "timeout_bars": exit at close if neither stop nor target is
            hit within that many bars after entry.
        bars: OHLC bars to walk forward through, sorted ascending by time,
            starting at or after signal["time"]. Must have columns
            {time, open, high, low, close}.
        round_trip_cost: cost to apply. Defaults to the gold constant in
            core.costs; pass an instrument-specific cost for other symbols.

    Returns:
        Outcome dict: {time, direction, entry, stop, target, exit_time,
        exit_price, result, gross, net, r_multiple}, where result is one
        of "stop", "target", "timeout".
    """
    direction = signal["direction"]
    entry = signal["entry"]
    stop = signal["stop"]
    target = signal["target"]
    timeout_bars = signal.get("timeout_bars")
    risk = abs(entry - stop)

    start_pos = bars["time"].searchsorted(signal["time"], side="right")

    exit_time = None
    exit_price = None
    result = None

    n_walked = 0
    for i in range(start_pos, len(bars)):
        bar = bars.iloc[i]
        n_walked += 1

        if direction == "long":
            hit_stop = bar["low"] <= stop
            hit_target = bar["high"] >= target
        else:
            hit_stop = bar["high"] >= stop
            hit_target = bar["low"] <= target

        if hit_stop:
            exit_time, exit_price, result = bar["time"], stop, "stop"
            break
        if hit_target:
            exit_time, exit_price, result = bar["time"], target, "target"
            break
        if timeout_bars is not None and n_walked >= timeout_bars:
            exit_time, exit_price, result = bar["time"], bar["close"], "timeout"
            break
    else:
        if len(bars) > start_pos:
            last = bars.iloc[-1]
            exit_time, exit_price, result = last["time"], last["close"], "timeout"
        else:
            exit_time, exit_price, result = signal["time"], entry, "timeout"

    if direction == "long":
        gross = exit_price - entry
    else:
        gross = entry - exit_price

    net = apply_cost(gross) if round_trip_cost is None else apply_cost(gross, round_trip_cost)
    r_multiple = (net / risk) if risk else 0.0

    return {
        "time": signal["time"],
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "target": target,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "result": result,
        "gross": gross,
        "net": net,
        "r_multiple": r_multiple,
    }


def simulate_all(signals, bars, round_trip_cost=None):
    """
    Run simulate_trade for each signal in signals against bars.

    Args:
        signals: list of signal dicts, see simulate_trade.
        bars: OHLC bars covering the full period spanned by signals.
        round_trip_cost: cost to apply, see simulate_trade.

    Returns:
        List of outcome records, one per signal.
    """
    return [simulate_trade(signal, bars, round_trip_cost) for signal in signals]

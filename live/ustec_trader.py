"""
Live execution of the H1/M5 market structure strategy on USTECm, demo
account only.

This is an execution wrapper around strats/market_structure_h1.py --
generate_signals() is imported and called unchanged. Nothing in this
file alters structure detection, zone construction, entry triggers,
stops, targets, or the 288 M5-bar timeout; those all live in the
strategy module and are reused as-is.

To guarantee the live bot produces the same signals the backtest does,
each cycle calls generate_signals() over the full accumulated history
(the same data/cache/*.pkl the backtest reads, extended forward with
newly closed bars each cycle) rather than a rolling window -- a rolling
window would let the strategy's internal BOS/active-swing state diverge
from what a full-history run would show.

Run:
    python live/ustec_trader.py            # dry run: no orders sent
    python live/ustec_trader.py --live      # places real orders on
                                             # whatever account MT5_DEMO_*
                                             # env vars point to (refuses
                                             # to run against a non-demo
                                             # account)

Credentials: MT5_DEMO_LOGIN / MT5_DEMO_PASSWORD / MT5_DEMO_SERVER env
vars only, via data.fetch's own connection helper. Never hardcoded.
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch
from core.costs import apply_cost
from strats import market_structure_h1 as strat

SYMBOL = "USTECm"
STRUCTURE_TF = "H1"
ENTRY_TF = "M5"
TIMEOUT_BARS = strat.ENTRY_TIMEOUT_M5_BARS  # 288, reused from the strategy module
RISK_PCT = 0.01
MAGIC = 20260915
POLL_SECONDS = 15
ORDER_DEVIATION_POINTS = 20

STATE_PATH = Path(__file__).parent / "state" / f"{SYMBOL}_h1m5_state.json"
LOG_PATH = Path(__file__).parent / "logs" / f"{SYMBOL}_h1m5_log.csv"

LOG_FIELDS = [
    "timestamp", "signal_fired", "direction", "entry", "stop", "target", "rr",
    "lots", "actual_risk_pct", "floor_forced", "skip_reason",
    "exit_time", "exit_price", "result", "gross", "net", "r_multiple",
]


# ---------------------------------------------------------------- connection

def connect(mt5):
    """Reuse data.fetch's own connect/login helper -- env vars only."""
    ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    fetch._connect(mt5)


def assert_demo_account(mt5):
    acct = mt5.account_info()
    if acct is None:
        raise RuntimeError("no account info after login")
    if acct.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        raise RuntimeError(
            f"refusing to trade: account {acct.login}@{acct.server} is not a demo "
            f"account (trade_mode={acct.trade_mode}). This bot only runs on demo."
        )
    return acct


# --------------------------------------------------------------------- bars

def _merge_bars(old_df, new_df):
    if new_df is None or len(new_df) == 0:
        return old_df
    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return merged


def load_base_bars():
    """Start from the full cached history -- same files the backtest reads."""
    bars = {}
    for tf in (STRUCTURE_TF, ENTRY_TF):
        try:
            bars[tf] = fetch.load_from_cache(SYMBOL, tf)
        except FileNotFoundError:
            bars[tf] = pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])
    return bars


def refresh_bars(mt5, bars):
    """Fetch and merge in any newly closed bars for both timeframes."""
    now = datetime.now()
    for tf in (STRUCTURE_TF, ENTRY_TF):
        last_time = bars[tf]["time"].max() if len(bars[tf]) else datetime(2018, 1, 1)
        new = fetch.fetch(SYMBOL, tf, last_time, now)
        bars[tf] = _merge_bars(bars[tf], new)
        fetch.save_to_cache(SYMBOL, tf, bars[tf])
    return bars


# -------------------------------------------------------------------- state

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"open": False}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# --------------------------------------------------------------------- cost

def current_round_trip_cost(mt5, symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info failed for {symbol}: {mt5.last_error()}")
    return info.spread * info.point


# ------------------------------------------------------------------ sizing

def compute_lots(mt5, symbol, entry, stop, risk_pct):
    info = mt5.symbol_info(symbol)
    acct = mt5.account_info()
    risk_distance = abs(entry - stop)

    value_per_lot = (risk_distance / info.trade_tick_size) * info.trade_tick_value
    target_risk_dollars = acct.equity * risk_pct

    raw_lots = target_risk_dollars / value_per_lot if value_per_lot else 0.0
    step = info.volume_step
    lots = math.floor(raw_lots / step) * step if step else raw_lots
    floor_forced = lots < info.volume_min
    if floor_forced:
        lots = info.volume_min
    lots = min(lots, info.volume_max)
    lots = round(lots, 8)

    actual_risk_dollars = lots * value_per_lot
    actual_risk_pct = 100.0 * actual_risk_dollars / acct.equity if acct.equity else float("nan")

    return lots, actual_risk_pct, floor_forced


# ----------------------------------------------------------------- filling

def _pick_filling(info, mt5):
    if info.filling_mode & mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    if info.filling_mode & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


# ---------------------------------------------------------------- execution

def open_position(mt5, symbol, signal, lots, dry_run):
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    is_long = signal["direction"] == "long"
    order_type = mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL
    price = tick.ask if is_long else tick.bid

    if dry_run:
        return {"retcode": "DRY_RUN", "order": None, "price": price}

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": signal["stop"],
        "tp": signal["target"],
        "deviation": ORDER_DEVIATION_POINTS,
        "magic": MAGIC,
        "comment": "ustec_h1m5",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _pick_filling(info, mt5),
    }
    result = mt5.order_send(request)
    return {"retcode": result.retcode, "order": result.order, "price": result.price,
            "position": getattr(result, "deal", None)}


def close_position_market(mt5, symbol, position, comment, dry_run):
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    is_long = position.type == mt5.POSITION_TYPE_BUY
    close_type = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY
    price = tick.bid if is_long else tick.ask

    if dry_run:
        return {"retcode": "DRY_RUN", "price": price}

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": close_type,
        "position": position.ticket,
        "price": price,
        "deviation": ORDER_DEVIATION_POINTS,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _pick_filling(info, mt5),
    }
    result = mt5.order_send(request)
    return {"retcode": result.retcode, "price": result.price}


def find_closing_deal(mt5, ticket):
    """
    Look up the deal that closed position `ticket`. Returns None if the
    position isn't closed yet (or history hasn't caught up).
    """
    deals = mt5.history_deals_get(position=ticket)
    if not deals:
        return None
    out_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
    if not out_deals:
        return None
    d = out_deals[-1]
    if d.reason == mt5.DEAL_REASON_SL:
        result = "stop"
    elif d.reason == mt5.DEAL_REASON_TP:
        result = "target"
    else:
        result = "timeout"
    return {"time": datetime.fromtimestamp(d.time), "price": d.price, "result": result}


# ------------------------------------------------------------------- logging

def log_row(row):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def _base_row(now):
    return {
        "timestamp": now.isoformat(), "signal_fired": False, "direction": "",
        "entry": "", "stop": "", "target": "", "rr": "", "lots": "",
        "actual_risk_pct": "", "floor_forced": False, "skip_reason": "",
        "exit_time": "", "exit_price": "", "result": "", "gross": "",
        "net": "", "r_multiple": "",
    }


# --------------------------------------------------------------------- core

def process_closed_bar(mt5, state, h1_df, m5_df, just_closed_time, dry_run):
    """
    Run one strategy evaluation for the M5 bar that just closed at
    just_closed_time, using only data known as of that bar (no bars after
    it are visible to generate_signals -- preserves the strategy's own
    no-look-ahead guarantee when the wrapper is catching up on more than
    one missed bar).
    """
    now = datetime.now()
    row = _base_row(now)

    h1_visible = h1_df[h1_df["time"] <= just_closed_time].reset_index(drop=True)
    m5_visible = m5_df[m5_df["time"] <= just_closed_time].reset_index(drop=True)

    # --- 1. manage an already-open position first ---
    if state.get("open"):
        ticket = state["ticket"]
        positions = mt5.positions_get(ticket=ticket) if not dry_run else ()
        still_open = bool(positions) if not dry_run else state.get("open", False)

        if not dry_run and not still_open:
            closing = find_closing_deal(mt5, ticket)
            if closing is not None:
                row.update(_close_row(state, closing["time"], closing["price"], closing["result"]))
                state = {"open": False}
                save_state(state)
                log_row(row)
                return state

        elapsed_bars = int((m5_visible["time"] > pd.Timestamp(state["entry_time"])).sum())
        if elapsed_bars >= TIMEOUT_BARS:
            if dry_run:
                exit_price = m5_visible.iloc[-1]["close"]
                row.update(_close_row(state, just_closed_time, exit_price, "timeout"))
                state = {"open": False}
            else:
                positions = mt5.positions_get(ticket=ticket)
                if positions:
                    pos = positions[0]
                    res = close_position_market(mt5, SYMBOL, pos, "ustec_h1m5_timeout", dry_run)
                    closing = find_closing_deal(mt5, ticket)
                    if closing is not None:
                        row.update(_close_row(state, closing["time"], closing["price"], "timeout"))
                    else:
                        row.update(_close_row(state, now, res["price"], "timeout"))
                    state = {"open": False}
                else:
                    closing = find_closing_deal(mt5, ticket)
                    result = closing["result"] if closing else "timeout"
                    price = closing["price"] if closing else state["entry"]
                    ts = closing["time"] if closing else now
                    row.update(_close_row(state, ts, price, result))
                    state = {"open": False}
            save_state(state)
            log_row(row)
            return state

        # still open, no timeout yet: nothing else to do this cycle
        log_row(row)
        return state

    # --- 2. no open position: look for a fresh signal on this exact bar ---
    signals = strat.generate_signals({STRUCTURE_TF: h1_visible, ENTRY_TF: m5_visible})
    matching = [s for s in signals if s["time"] == just_closed_time]

    if not matching:
        log_row(row)
        return state

    signal = matching[0]
    row["signal_fired"] = True
    row["direction"] = signal["direction"]
    row["entry"] = signal["entry"]
    row["stop"] = signal["stop"]
    row["target"] = signal["target"]
    risk = abs(signal["entry"] - signal["stop"])
    reward = abs(signal["target"] - signal["entry"])
    row["rr"] = (reward / risk) if risk else ""

    # state["open"] is False here by construction, but a signal firing
    # while a position is open (shouldn't happen given the branch above,
    # kept as a defensive guard) is still a skip, not a second entry.
    if state.get("open"):
        row["skip_reason"] = "position_already_open"
        log_row(row)
        return state

    lots, actual_risk_pct, floor_forced = compute_lots(mt5, SYMBOL, signal["entry"], signal["stop"], RISK_PCT)
    row["lots"] = lots
    row["actual_risk_pct"] = actual_risk_pct
    row["floor_forced"] = floor_forced

    result = open_position(mt5, SYMBOL, signal, lots, dry_run)
    order_failed = (not dry_run) and result["retcode"] != mt5.TRADE_RETCODE_DONE
    if order_failed:
        row["skip_reason"] = f"order_send_failed:{result['retcode']}"
        log_row(row)
        return state

    cost = current_round_trip_cost(mt5, SYMBOL)
    if dry_run:
        ticket = None
    else:
        # Look up the actual position ticket rather than assuming it
        # equals the order ticket (not guaranteed across account types).
        positions = mt5.positions_get(symbol=SYMBOL)
        ours = [p for p in positions if p.magic == MAGIC]
        ticket = ours[-1].ticket if ours else result["order"]

    state = {
        "open": True,
        "ticket": ticket,
        "direction": signal["direction"],
        "entry": result["price"],
        "stop": signal["stop"],
        "target": signal["target"],
        "entry_time": just_closed_time.isoformat(),
        "signal_time": signal["time"].isoformat(),
        "round_trip_cost": cost,
        "dry_run": dry_run,
    }
    save_state(state)
    log_row(row)
    return state


def _close_row(state, exit_time, exit_price, result):
    entry = state["entry"]
    risk = abs(entry - state["stop"])
    gross = (exit_price - entry) if state["direction"] == "long" else (entry - exit_price)
    net = apply_cost(gross, state.get("round_trip_cost", 0.0))
    r_multiple = (net / risk) if risk else 0.0
    return {
        "exit_time": pd.Timestamp(exit_time).isoformat(),
        "exit_price": exit_price,
        "result": result,
        "gross": gross,
        "net": net,
        "r_multiple": r_multiple,
    }


# --------------------------------------------------------------------- main

def run_forever(dry_run):
    import MetaTrader5 as mt5

    connect(mt5)
    acct = assert_demo_account(mt5)
    print(f"connected: {acct.login}@{acct.server}  demo={acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO}  "
          f"dry_run={dry_run}")

    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"symbol_select failed for {SYMBOL}: {mt5.last_error()}")

    bars = load_base_bars()
    bars = refresh_bars(mt5, bars)
    state = load_state()

    last_processed_m5_time = None
    if len(bars[ENTRY_TF]):
        last_processed_m5_time = bars[ENTRY_TF]["time"].max()
        print(f"starting caught up through M5 bar {last_processed_m5_time}")

    while True:
        bars = refresh_bars(mt5, bars)
        m5_all = bars[ENTRY_TF]

        if last_processed_m5_time is None:
            new_bars = m5_all
        else:
            new_bars = m5_all[m5_all["time"] > last_processed_m5_time]

        for _, bar in new_bars.iterrows():
            just_closed_time = bar["time"]
            state = process_closed_bar(mt5, state, bars[STRUCTURE_TF], bars[ENTRY_TF], just_closed_time, dry_run)
            last_processed_m5_time = just_closed_time

        time.sleep(POLL_SECONDS)


def run_once_dry():
    """Single-cycle smoke test: connect, evaluate the latest closed bar,
    log the result, never sends an order. Used to verify the wiring
    before starting the persistent loop."""
    import MetaTrader5 as mt5

    connect(mt5)
    acct = assert_demo_account(mt5)
    print(f"connected: {acct.login}@{acct.server}  demo=True  dry_run=True (single cycle)")

    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"symbol_select failed for {SYMBOL}: {mt5.last_error()}")

    bars = load_base_bars()
    bars = refresh_bars(mt5, bars)
    state = load_state()

    just_closed_time = bars[ENTRY_TF]["time"].max()
    print(f"evaluating just-closed M5 bar: {just_closed_time}")
    state = process_closed_bar(mt5, state, bars[STRUCTURE_TF], bars[ENTRY_TF], just_closed_time, dry_run=True)
    print("cycle complete. state:", state)
    print(f"log written to {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="place real orders (demo account only)")
    parser.add_argument("--once", action="store_true", help="single dry-run cycle, then exit")
    args = parser.parse_args()

    if args.once:
        run_once_dry()
    else:
        run_forever(dry_run=not args.live)

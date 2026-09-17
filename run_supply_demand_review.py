"""
Follow-up to run_supply_demand.py addressing a code review:

1. Survivorship bias -- run Model A (plain_touch, every touch is a
   trade) and Model B (engulf, confirmation required) separately.
2. One-bar look-ahead in the bias/target filters -- fixed at the
   source in strats/supply_demand_mtf.py (usable_time = bos_time +
   one bar duration). This script's Model B numbers are the
   post-fix numbers; compare against the run_supply_demand.py
   report from before the fix to see how much it moved.
3. A real fit/validate split: fit on the first 6 months only, sweep
   stop_buffer x entry_trigger x target_mode, rank by expectancy in R
   among combos with >=30 trades, take the top 5, run them unchanged
   on the last 3 months.

TUNING DISCLOSURE: stop_buffer's default (3.0) and the single-candle
zone definition were both chosen to reproduce the trader's stated 2-4
point stops from their real examples. That's fitting to their
description of the strategy, not an independently derived parameter --
said here explicitly per their request.
"""

from datetime import datetime

import pandas as pd

from core.simulate import simulate_all
from data import fetch
from strats.supply_demand_mtf import prepare_zones, signals_from_zones

SYMBOL = "XAUUSDm"
ROUND_TRIP_COST_XAU = 0.40


def _load(tf):
    return fetch.load_from_cache(SYMBOL, tf)


def _stats(trades, label):
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0, "win_pct": 0.0, "avg_r": 0.0, "net": 0.0}
    wins = sum(1 for t in trades if t["net"] > 0)
    return {
        "label": label,
        "n": n,
        "win_pct": 100.0 * wins / n,
        "avg_r": sum(t["r_multiple"] for t in trades) / n,
        "net": sum(t["net"] for t in trades),
    }


def _print_stats(s):
    print(f"{s['label']:<45} n={s['n']:<6} win%={s['win_pct']:<7.1f} avgR={s['avg_r']:<8.3f} net={s['net']:<10.2f}")


def part1_model_comparison(bars, m1, split_date, data_start, data_end):
    print("\n=== PART 1: Model A (plain_touch) vs Model B (engulf), full 9 months ===")
    print("(Model A includes every losing touch that never confirmed -- Model B doesn't.)\n")

    zone_base = prepare_zones(bars)

    for trigger, label in [("plain_touch", "Model A: plain_touch"), ("engulf", "Model B: engulf")]:
        sigs = signals_from_zones(zone_base, entry_trigger=trigger, stop_buffer=3.0, target_mode="nearest_zone")
        trades = simulate_all(sigs, m1, round_trip_cost=ROUND_TRIP_COST_XAU)
        train = [t for t in trades if t["time"] < split_date]
        rest = [t for t in trades if t["time"] >= split_date]
        _print_stats(_stats(trades, f"{label} -- full 9mo"))
        _print_stats(_stats(train, f"{label} -- first half"))
        _print_stats(_stats(rest, f"{label} -- second half"))
        print()


def part3_sweep(bars_fit, bars_full, m1, fit_start, fit_end, validate_start, validate_end):
    print("\n=== PART 3: fit on first 6 months, sweep, validate top 5 on last 3 months ===\n")

    zone_base_fit = prepare_zones(bars_fit)
    zone_base_full = prepare_zones(bars_full)  # for validate: sees all history up to data_end

    stop_buffers = [2, 4, 6, 8, 10]
    entry_triggers = ["engulf", "pin", "plain_touch"]
    target_specs = [("nearest_zone", None), ("fixed", 25), ("fixed", 35), ("fixed", 50)]

    results = []
    for sb in stop_buffers:
        for trig in entry_triggers:
            for tmode, tpts in target_specs:
                sigs = signals_from_zones(zone_base_fit, entry_trigger=trig, stop_buffer=sb,
                                           target_mode=tmode, target_fixed_pts=tpts)
                sigs = [s for s in sigs if fit_start <= s["time"] < fit_end]
                trades = simulate_all(sigs, m1, round_trip_cost=ROUND_TRIP_COST_XAU)
                n = len(trades)
                avg_r = sum(t["r_multiple"] for t in trades) / n if n else float("-inf")
                win_pct = 100.0 * sum(1 for t in trades if t["net"] > 0) / n if n else 0.0
                target_label = tmode if tmode == "nearest_zone" else f"fixed{tpts}"
                results.append({
                    "stop_buffer": sb, "entry_trigger": trig, "target_mode": target_label,
                    "n": n, "win_pct": win_pct, "avg_r": avg_r,
                })

    all_results = pd.DataFrame(results)
    print(f"total combos: {len(all_results)}, with >=30 trades: {(all_results['n'] >= 30).sum()}")

    eligible = all_results[all_results["n"] >= 30].copy()
    eligible = eligible.sort_values("avg_r", ascending=False)
    top5 = eligible.head(5)

    print("\n--- top 5 by fit-period expectancy (avg R), min 30 trades ---")
    print(top5.to_string(index=False))

    print("\n--- fit vs validate, same params, unchanged ---")
    for _, row in top5.iterrows():
        tmode = row["target_mode"]
        if tmode == "nearest_zone":
            target_mode, target_fixed_pts = "nearest_zone", None
        else:
            target_mode, target_fixed_pts = "fixed", int(tmode.replace("fixed", ""))

        sigs_val = signals_from_zones(zone_base_full, entry_trigger=row["entry_trigger"], stop_buffer=row["stop_buffer"],
                                       target_mode=target_mode, target_fixed_pts=target_fixed_pts)
        sigs_val = [s for s in sigs_val if validate_start <= s["time"] < validate_end]
        trades_val = simulate_all(sigs_val, m1, round_trip_cost=ROUND_TRIP_COST_XAU)
        val_stats = _stats(trades_val, "validate")

        label = f"sb={row['stop_buffer']} trig={row['entry_trigger']} tgt={tmode}"
        print(f"\n{label}")
        print(f"  fit:      n={row['n']:<6} win%={row['win_pct']:<7.1f} avgR={row['avg_r']:<8.3f}")
        print(f"  validate: n={val_stats['n']:<6} win%={val_stats['win_pct']:<7.1f} avgR={val_stats['avg_r']:<8.3f} net={val_stats['net']:<10.2f}")

    return all_results


def run():
    h1, m30, m15, m1 = _load("H1"), _load("M30"), _load("M15"), _load("M1")
    bars = {"H1": h1, "M30": m30, "M15": m15}

    data_start = m1["time"].min()
    data_end = m1["time"].max()
    split_date = data_start + (data_end - data_start) / 2

    part1_model_comparison(bars, m1, split_date, data_start, data_end)

    fit_start = data_start
    fit_end = data_start + pd.DateOffset(months=6)
    validate_start = fit_end
    validate_end = data_end

    bars_fit = {tf: bars[tf][bars[tf]["time"] < fit_end].reset_index(drop=True) for tf in bars}

    print(f"\nfit window:      {fit_start} .. {fit_end}")
    print(f"validate window: {validate_start} .. {validate_end}")

    sweep_results = part3_sweep(bars_fit, bars, m1, fit_start, fit_end, validate_start, validate_end)
    sweep_results.to_csv("results/supply_demand_sweep_all_combos.csv", index=False)
    print("\nfull sweep grid written to results/supply_demand_sweep_all_combos.csv")


if __name__ == "__main__":
    run()

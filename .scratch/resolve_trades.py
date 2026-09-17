import pandas as pd
from datetime import datetime

df = pd.read_pickle("F:/f/Gold-Lab/.scratch/xau_m1_sep15.pkl").reset_index(drop=True)

trades = [
    {"id": 1, "dir": "short", "entry_time": datetime(2026, 9, 15, 2, 50), "entry": 4316.082, "sl": 4319.654, "tp": 4283.505},
    {"id": 2, "dir": "long", "entry_time": datetime(2026, 9, 15, 9, 15), "entry": 4263.927, "sl": 4260.926, "tp": 4295.747},
    {"id": 3, "dir": "long", "entry_time": datetime(2026, 9, 15, 14, 50), "entry": 4275.644, "sl": 4273.501, "tp": 4301.221},
]

for t in trades:
    start_pos = df["time"].searchsorted(t["entry_time"], side="left")
    print(f"\n--- Trade {t['id']} ({t['dir']}) entry={t['entry']} sl={t['sl']} tp={t['tp']} at {t['entry_time']} UTC ---")
    print("nearest bars around entry:")
    print(df.iloc[max(0, start_pos - 2):start_pos + 3].to_string())

    result = None
    for i in range(start_pos, len(df)):
        bar = df.iloc[i]
        if t["dir"] == "long":
            hit_sl = bar["low"] <= t["sl"]
            hit_tp = bar["high"] >= t["tp"]
        else:
            hit_sl = bar["high"] >= t["sl"]
            hit_tp = bar["low"] <= t["tp"]

        if hit_sl and hit_tp:
            result = ("AMBIGUOUS - same bar touches both SL and TP", bar["time"], bar)
            break
        if hit_sl:
            result = ("SL", bar["time"], bar)
            break
        if hit_tp:
            result = ("TP", bar["time"], bar)
            break

    if result is None:
        print("Neither SL nor TP hit within available data window.")
    else:
        label, when, bar = result
        bars_walked = i - start_pos + 1
        print(f"RESULT: {label} at {when} UTC (bar #{bars_walked} after entry)")
        print(bar[["time", "open", "high", "low", "close"]].to_string())

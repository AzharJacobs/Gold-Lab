# Gold-Lab

A lab for testing gold (XAUUSD) trading strategies against MT5 historical data.

## Structure

```
Gold-Lab/
├── data/
│   ├── cache/        cached bars, one .pkl per symbol/timeframe
│   └── fetch.py       pulls M15/H1/H4/D1 from MT5, closed bars only
├── core/
│   ├── costs.py        ROUND_TRIP_COST and cost application
│   ├── simulate.py      walks a signal forward through bars to an outcome
│   └── validate.py      splits trades by date, reports each half separately
├── strats/             one file per strategy
├── results/            output of run.py
├── run.py              loads a strat, runs it, prints the report
└── README.md
```

## Rules

- **Costs are always applied.** Every simulated trade goes through `core/costs.py`. There is no "gross" number that gets reported on its own.
- **No tuning after seeing validate.** Parameters, filters, and entry/exit logic are locked in before the validate split is looked at. If validate performance changes your mind about the strategy's logic, that logic gets thrown out — it doesn't get adjusted.
- **No filters added to fix losing periods.** If a strategy loses money in some stretch of history, the fix is not a filter that excludes that stretch. Either the strategy is robust across periods or it isn't.
- **Never report a blended number alone.** Train and validate (and the last-6-months window) are always shown side by side, per `core/validate.py`. A single combined win-rate or net-R number is not a valid report.

## Strategy interface

A file in `strats/` exports exactly one function: it takes bars and returns a list of signals. Nothing else — no side effects, no I/O, no parameter tuning based on outcomes.

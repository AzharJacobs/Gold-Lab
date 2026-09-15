"""
Trading costs. Applied to every simulated trade, no exceptions.
"""

ROUND_TRIP_COST = 0.34


def apply_cost(gross_result: float, round_trip_cost: float = ROUND_TRIP_COST) -> float:
    """
    Apply the round-trip cost to a gross trade result, returning the net result.

    Args:
        gross_result: trade P&L before costs, in the same units as round_trip_cost.
        round_trip_cost: cost to subtract. Defaults to the gold constant;
            pass an instrument-specific cost for other symbols.

    Returns:
        Net trade result after subtracting round_trip_cost.
    """
    return gross_result - round_trip_cost

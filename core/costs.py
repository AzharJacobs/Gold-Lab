"""
Trading costs. Applied to every simulated trade, no exceptions.
"""

ROUND_TRIP_COST = 0.34


def apply_cost(gross_result: float) -> float:
    """
    Apply the round-trip cost to a gross trade result, returning the net result.

    Args:
        gross_result: trade P&L before costs, in the same units as ROUND_TRIP_COST.

    Returns:
        Net trade result after subtracting ROUND_TRIP_COST.
    """
    return gross_result - ROUND_TRIP_COST

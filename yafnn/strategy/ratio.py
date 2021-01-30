import numpy as np


def neg_sharpe_ratio(allocations, prices, rfr=0.0):
    """Annualized reward to risk variability.

    allocations : 2d-array
        Percent allocations (0-1) that each individual
        stock contributes. Must sum to one.

    prices : 2d-array
        Normalized set of prices; each column is a
        distinct stock

    rfr : float, default=0.0
        Risk-free return rate, e.g. 10-year
        U.S. Treasury bond

    Returns : float
        Negative Sharpe ratio
    """
    # Allocations at each timestamp must sum to one
    if allocations.ndim > 1:
        np.testing.assert_almost_equal(
            np.sum(allocations, axis=1), np.repeat(1, allocations.shape[0])
        )
    else:
        np.testing.assert_almost_equal(
            np.sum(allocations), np.repeat(1, allocations.shape[0])
        )

    # Calculate portfolio position values
    values_all = prices * allocations
    values_timestamp = values_all.sum(axis=1)

    # Determine daily returns from entire portfolio
    returns = values_timestamp[1:] / values_timestamp[:-1]

    k = np.sqrt(252)  # Annualize by accounting for yearly trading days
    s = np.mean(returns - rfr) / np.std(returns)
    return -1 * k * s

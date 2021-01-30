import numpy as np

from yafnn.strategy.ratio import neg_sharpe_ratio
from yafnn.strategy.allocation import AllocationStrategy


def test_allocation_strategy_opt_allocs():
    """Ensure 100% allocation to a steadily increasing
    asset as compared to a highly risky one.
    """
    prices = np.array([[10, 10], [11, 15], [12, 5], [13, 10]])
    allocs = AllocationStrategy.opt_allocs(prices, neg_sharpe_ratio)
    np.testing.assert_almost_equal(allocs, [1, 0])

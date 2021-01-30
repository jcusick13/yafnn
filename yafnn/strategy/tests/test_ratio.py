import numpy as np
import pytest

from yafnn.strategy.ratio import neg_sharpe_ratio


def test_neg_sharpe_ratio_allocations_sum_to_one():
    allocs = np.array([[0.5, 0.5], [0.5, 0.6]])
    prices = np.array([[1, 1], [1, 1]])
    with pytest.raises(AssertionError):
        neg_sharpe_ratio(allocs, prices)


def test_neg_sharpe_ratio_single_holding():
    """Ensure basic ratio is calculated correctly"""
    allocs = np.array([[1], [1], [1]])
    prices = np.array([[2], [4], [16]])
    prices_norm = prices / prices[0]
    sr = neg_sharpe_ratio(allocs, prices_norm)

    assert sr == -3 * np.sqrt(252)

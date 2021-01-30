import numpy as np
from scipy import optimize


class AllocationStrategy:
    @staticmethod
    def opt_allocs(prices, fun):
        """Determine optimal holdings allocation

        prices : 2d-array
            Asset values (cols) at each timestamp (rows)
        """
        prices_norm = prices / prices[0, :]
        n_holdings = prices.shape[1]
        even_alloc = np.full(n_holdings, (1 / n_holdings))
        bounds = np.full(
            (n_holdings, 2), (0, 1)
        )  # Each holding is a fractional part of portfolio
        constraints = {'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)}
        res = optimize.minimize(
            fun,
            x0=even_alloc,
            args=(prices_norm,),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
        )
        if res.success:
            return res.x
        return np.full(n_holdings, -99)

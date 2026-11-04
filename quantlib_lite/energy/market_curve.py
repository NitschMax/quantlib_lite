from dataclasses import dataclass

import numpy as np

from quantlib_lite.stochastic_models import ForwardCurveModel


@dataclass(frozen=True)
class MarketCurve:
    t: float
    prices: dict[int, float]

    @classmethod
    def from_model(cls, model, t, state, months, points_per_month=30 ):
        if np.ndim(state) != 1: raise ValueError("State is not a vector")
        if not isinstance(model, ForwardCurveModel): raise TypeError("Invalid model")
        prices = {int(m): float(price) for m, price in zip(months, model.monthly_forward(t, months, state, points_per_month)) }
        return cls(t, prices)

    def price(self, month) -> float:
        # First check if month is valid key in prices, then return corresponding value
        if not (month in self.prices):
            raise KeyError(f"Invalid month {month}")
        return self.prices[month]

    def months(self)  -> list[int]:
        # Return a sorted list of all the keys in prices
        return sorted(self.prices.keys())

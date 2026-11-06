from dataclasses import dataclass
import numpy as np

from quantlib_lite.stochastic_models import ForwardCurveModel
from . import calendar as cld


@dataclass(frozen=True)
class MarketCurve:
    t: float
    prices: dict[int, float]

    @classmethod
    def from_model(cls, model, t, state, months):
        if np.ndim(state) != 1: raise ValueError("State is not a vector")
        if not isinstance(model, ForwardCurveModel): raise TypeError("Invalid model")

        months = np.array(months, int)
        prices = {int(m): float(price) for m, price in zip(months, model.average_forward(t, cld.month_start(months), cld.month_end(months), state, cld.days_in_month(months))) }
        return cls(t, prices)

    def price(self, month) -> float:
        # First check if month is valid key in prices, then return corresponding value
        if not (month in self.prices):
            raise KeyError(f"Invalid month {month} is not part of the prices. Available keys are {self.prices.keys()}")
        return self.prices[month]

    def months(self)  -> list[int]:
        # Return a sorted list of all the keys in prices
        return sorted(self.prices.keys())

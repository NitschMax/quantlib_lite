from dataclasses import dataclass

from quantlib_lite.stochastic_models import ForwardCurveModel


@dataclass(frozen=True)
class MarketCurve:

    @classmethod
    def from_model(cls, model, t, state, months, points_per_month):
        if not isinstance(model, ForwardCurveModel): raise TypeError("Invalid model")
        return model.monthly_forward(t, months, state, points_per_month)

from quantlib_lite.stochastic_models.stochastic_model import StochasticModel
from quantlib_lite.payoff.payoff import Payoff
from quantlib_lite.risk_measure.risk_measure import RiskMeasure

class Pricer:
    def __init__(self, model, payoff, risk):
        if isinstance(model, StochasticModel):
            self.__model = model
        else:
            raise TypeError('model must be an instance of StochasticModel.')

        if isinstance(payoff, Payoff):
            self.__payoff = payoff
        else:
            raise TypeError('payoff must be an instance of Payoff.')

        if isinstance(risk, RiskMeasure):
            self.__risk = risk
        else:
            raise TypeError('risk must be an instance of RiskMeasure.')

    @property
    def model(self):
        return self.__model

    @property
    def payoff(self):
        return self.__payoff

    @property
    def risk(self):
        return self.__risk

    def price_single_path(self, T, steps):
        path = self.model.sample_path(T, steps)
        return self.payoff.evaluate(path)

    def price(self, T, steps, samples=1000):
        prices = [self.price_single_path(T, steps) for _ in range(samples)]
        return self.risk.evaluate(prices)


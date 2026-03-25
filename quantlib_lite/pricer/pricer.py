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
            self._payoff = payoff
        else:
            raise TypeError('payoff must be an instance of Payoff.')

        if isinstance(risk, RiskMeasure):
            self._risk = risk
        else:
            raise TypeError('risk must be an instance of RiskMeasure.')

        self._cache = {}    # The cache is used only to save simulation paths which can be used independent of changes in payoff and risk

    @property
    def model(self):
        return self.__model

    @property
    def payoff(self):
        return self._payoff

    @property
    def risk(self):
        return self._risk

    def set_payoff(self, payoff):
        if isinstance(payoff, Payoff):
            self._payoff = payoff
        else:
            raise TypeError('payoff must be an instance of Payoff.')

    def set_risk(self, risk):
        if isinstance(risk, RiskMeasure):
            self._risk = risk
        else:
            raise TypeError('risk must be an instance of RiskMeasure.')

    def empty_cache(self):
        self._cache = {}

    def simulation_key(self, T, steps):
        return (self.model, T, steps)

    def price(self, T, steps, samples=1000):
        key = self.simulation_key(T, steps)
        paths = self._cache.get(key, [])

        len_diff = int(samples) - len(paths)
        if len_diff > 0:
            paths.extend([self.model.sample_path(T, steps) for _ in range(len_diff)])
            self._cache[key] = paths
        
        prices = [self.payoff.evaluate(path) for path in paths[:samples]]
        return self.risk.evaluate(prices)


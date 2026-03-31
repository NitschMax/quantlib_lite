from quantlib_lite.stochastic_models import StochasticModel
from quantlib_lite.payoff import Payoff
from .portfolio import Portfolio
import numpy as np

class Hedger():
    def __init__(self, model, payoff, hedgingstrategy):
        if isinstance(model, StochasticModel):
            self.__model = model
        else:
            raise TypeError('model must be an instance of StochasticModel.')
        
        if isinstance(payoff, Payoff):
            self.__payoff = payoff
        else:
            raise TypeError('payoff must be an instance of Payoff.')

        self.hedgingstrategy = hedgingstrategy

    @property
    def model(self):
        return self.__model

    @property
    def payoff(self):
        return self.__payoff

    def run(self, T, r, steps):
        K = self.payoff.K
        sigma = self.model.sigma
        dt = self.model.dt(T, steps)

        path = self.model.sample_path(T, steps)
        S_0 = path.values[0]

        initial_value = self.hedgingstrategy.compute_initial_value(S_0, T, r, sigma, K)
        portfolio = Portfolio(cash_value = initial_value, asset_count = 0)
        
        for idx, (t, S_t) in enumerate(path[:-1]):
            a_t = self.hedgingstrategy.compute_delta(t, S_t, T, r, sigma, K)
            portfolio.update(a_t, S_t)
            portfolio.cash_value *= np.exp(r * dt)

        portfolio_value = portfolio.value_at_price_S(path.values[-1])
        payoff_value = self.payoff.evaluate(path)
        error = portfolio_value - payoff_value
        return portfolio_value, error


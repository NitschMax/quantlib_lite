from quantlib_lite.simulation_engine import SimulationEngine
from quantlib_lite.payoff import Payoff
from .portfolio import Portfolio
import numpy as np

class Hedger():
    def __init__(self, engine, payoff, hedgingstrategy):
        if isinstance(engine, SimulationEngine):
            self._engine = engine
        else:
            raise TypeError('engine must be an instance of SimulationEngine.')
        
        if isinstance(payoff, Payoff):
            self._payoff = payoff
        else:
            raise TypeError('payoff must be an instance of Payoff.')

        self.hedgingstrategy = hedgingstrategy

    @property
    def engine(self):
        return self._engine

    @property
    def payoff(self):
        return self._payoff

    def run(self, r, n_paths):
        K = self.payoff.K
        sigma = self.engine.model.sigma
        dt = self.engine.model.dt(T, steps)
        exp_r_dt = np.exp(r * dt)

        paths = self.engine.simulate(n_paths)
        portfolio_values, errors, S_T_array, payoff_values = np.empty((len(paths), 4))

        for idx, path in enumerate(paths):
            S_0 = path.values[0]

            initial_value = self.hedgingstrategy.compute_initial_value(S_0, T, r, sigma, K, self.payoff)
            portfolio = Portfolio(cash_value = initial_value, asset_count = 0)
            
            for idx, (t, S_t) in enumerate(path[:-1]):
                a_t = self.hedgingstrategy.compute_a_t(t, S_t, T, r, sigma, K, self.payoff)
                portfolio.update(a_t, S_t)
                portfolio.cash_value *= exp_r_dt

            portfolio_values[idx] = portfolio.value_at_price_S(path.values[-1])
            payoff_values[idx] = self.payoff.evaluate(path)
            errors[idx] = portfolio_value - payoff_value
            S_T_array[idx] = path.values[-1]
        return portfolio_values, errors, S_T_array, payoff_values


from quantlib_lite.simulation_engine import SimulationEngine
from quantlib_lite.payoff import Payoff
from quantlib_lite.risk_measure import RiskMeasure

class Pricer:
    def __init__(self, engine, payoff, risk):
        if isinstance(engine, SimulationEngine):
            self._engine = engine
        else:
            raise TypeError('engine must be an instance of SimulationEngine.')

        if isinstance(payoff, Payoff):
            self._payoff = payoff
        else:
            raise TypeError('payoff must be an instance of Payoff.')

        if isinstance(risk, RiskMeasure):
            self._risk = risk
        else:
            raise TypeError('risk must be an instance of RiskMeasure.')

    @property
    def engine(self):
        return self._engine

    @property
    def payoff(self):
        return self._payoff

    @property
    def risk(self):
        return self._risk

    @engine.setter
    def engine(self, engine):
        if isinstance(engine, SimulationEngine):
            self._engine = engine
        else:
            raise TypeError('engine must be an instance of SimulationEngine.')

    @payoff.setter
    def payoff(self, payoff):
        if isinstance(payoff, Payoff):
            self._payoff = payoff
        else:
            raise TypeError('payoff must be an instance of Payoff.')

    @risk.setter
    def risk(self, risk):
        if isinstance(risk, RiskMeasure):
            self._risk = risk
        else:
            raise TypeError('risk must be an instance of RiskMeasure.')

    def price(self, samples=1000):
        paths = self.engine.simulate(samples)
        prices = [self.payoff.evaluate(path) for path in paths]
        return self.risk.evaluate(prices)


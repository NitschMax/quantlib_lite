import numpy as np

from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall
from quantlib_lite.risk_measure import RiskFree
from quantlib_lite import Pricer
from quantlib_lite import SimulationEngine


def test_pricer_runs():
    model = GBM(mu=0.0, sigma=0.1)
    engine = SimulationEngine(model, T=1.0, steps=50, seed=1)
    payoff = EuropeanCall(K=1.0)
    risk = RiskFree()

    pricer = Pricer(engine, payoff, risk)

    price = pricer.price(samples=100)

    assert isinstance(price, float)


def test_price_positive():
    model = GBM(mu=0.0, sigma=0.2)
    engine = SimulationEngine(model, T=1.0, steps=50, seed=1)
    payoff = EuropeanCall(K=0.5)
    risk = RiskFree()

    pricer = Pricer(engine, payoff, risk)

    price = pricer.price(samples=200)

    assert price >= 0.0

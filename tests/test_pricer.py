import numpy as np

from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall
from quantlib_lite.risk_measure import RiskFree
from quantlib_lite import Pricer


def test_pricer_runs():
    model = GBM(mu=0.0, sigma=0.1)
    payoff = EuropeanCall(K=1.0)
    risk = RiskFree()

    pricer = Pricer(model, payoff, risk)

    price = pricer.price(T=1.0, steps=50, samples=100)

    assert isinstance(price, float)


def test_price_positive():
    model = GBM(mu=0.0, sigma=0.2)
    payoff = EuropeanCall(K=0.5)
    risk = RiskFree()

    pricer = Pricer(model, payoff, risk)

    price = pricer.price(T=1.0, steps=50, samples=200)

    assert price >= 0.0

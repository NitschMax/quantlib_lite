import numpy as np
import pytest

from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall
from quantlib_lite.hedger import DeltaHedgingStrategy, Hedger


def setup_hedger(S0=1.0, mu=0.05, sigma=0.2, K=1.0):
    model = GBM(mu=mu, sigma=sigma)
    payoff = EuropeanCall(K=K)
    strategy = DeltaHedgingStrategy()
    hedger = Hedger(model, payoff, strategy)
    return hedger


def test_hedger_runs():
    hedger = setup_hedger()
    T, r, steps = 1.0, 0.02, 10

    result = hedger.run(T, r, steps)

    assert isinstance(result, tuple)
    assert len(result) >= 2


def test_zero_steps_edge_case():
    hedger = setup_hedger()
    T, r, steps = 1.0, 0.02, 1

    portfolio_value, error, *_ = hedger.run(T, r, steps)

    assert np.isfinite(portfolio_value)
    assert np.isfinite(error)


def test_error_decreases_with_steps():
    hedger = setup_hedger()
    T, r = 1.0, 0.02

    steps_low = 3
    steps_high = 200
    n_paths = 200

    errors_low = []
    errors_high = []

    for _ in range(n_paths):
        _, err_low, *_ = hedger.run(T, r, steps_low)
        _, err_high, *_ = hedger.run(T, r, steps_high)

        errors_low.append(err_low)
        errors_high.append(err_high)

    std_low = np.std(errors_low)
    std_high = np.std(errors_high)

    assert std_high < std_low


def test_mean_error_close_to_zero():
    hedger = setup_hedger()
    T, r, steps = 1.0, 0.02, 200
    n_paths = 500

    errors = []

    for _ in range(n_paths):
        _, error, *_ = hedger.run(T, r, steps)
        errors.append(error)

    mean_error = np.mean(errors)

    assert abs(mean_error) < 1e-2


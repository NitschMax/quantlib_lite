import numpy as np
import pytest

from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall, EuropeanPut
from quantlib_lite.hedger import DeltaHedgingStrategy, Hedger


def setup_hedger(payoff_cls, mu=0.05, sigma=0.2, K=1.0):
    model = GBM(mu=mu, sigma=sigma)
    payoff = payoff_cls(K=K)
    strategy = DeltaHedgingStrategy()
    hedger = Hedger(model, payoff, strategy)
    return hedger


@pytest.mark.parametrize("payoff_cls", [EuropeanCall, EuropeanPut])
def test_hedger_runs(payoff_cls):
    hedger = setup_hedger(payoff_cls)
    T, r, steps = 1.0, 0.02, 10

    result = hedger.run(T, r, steps)

    assert isinstance(result, tuple)
    assert len(result) >= 2


@pytest.mark.parametrize("payoff_cls", [EuropeanCall, EuropeanPut])
def test_zero_steps_edge_case(payoff_cls):
    hedger = setup_hedger(payoff_cls)
    T, r, steps = 1.0, 0.02, 1

    portfolio_value, error, *_ = hedger.run(T, r, steps)

    assert np.isfinite(portfolio_value)
    assert np.isfinite(error)


@pytest.mark.parametrize("payoff_cls", [EuropeanCall, EuropeanPut])
def test_error_decreases_with_steps(payoff_cls):
    hedger = setup_hedger(payoff_cls)
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


@pytest.mark.parametrize("payoff_cls", [EuropeanCall, EuropeanPut])
def test_mean_error_close_to_zero(payoff_cls):
    hedger = setup_hedger(payoff_cls)
    T, r, steps = 1.0, 0.02, 200
    n_paths = 500

    errors = []

    for _ in range(n_paths):
        _, error, *_ = hedger.run(T, r, steps)
        errors.append(error)

    mean_error = np.mean(errors)

    assert abs(mean_error) < 1e-2


import numpy as np
import pytest

from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall, EuropeanPut
from quantlib_lite.hedger import DeltaHedgingStrategy, Hedger
from quantlib_lite.simulation_engine import SimulationEngine


def setup_hedger(payoff_cls, mu=0.05, sigma=0.2, K=1.0):
    model = GBM(mu=mu, sigma=sigma)
    T, steps = 1.0, 10
    engine = SimulationEngine(model, T, steps)
    payoff = payoff_cls(K=K)
    strategy = DeltaHedgingStrategy()
    hedger = Hedger(engine, payoff, strategy)
    return hedger


@pytest.mark.parametrize("payoff_cls", [EuropeanCall, EuropeanPut])
def test_hedger_runs(payoff_cls):
    r = 0.02
    hedger = setup_hedger(payoff_cls)

    result = hedger.run(r, n_paths=1)

    assert isinstance(result, tuple)
    assert len(result) >= 2


@pytest.mark.parametrize("payoff_cls", [EuropeanCall, EuropeanPut])
def test_zero_steps_edge_case(payoff_cls):
    r = 0.02
    hedger = setup_hedger(payoff_cls)

    portfolio_value, error, *_ = hedger.run(r, n_paths=1)

    assert np.isfinite(portfolio_value)
    assert np.isfinite(error)


@pytest.mark.parametrize("payoff_cls", [EuropeanCall, EuropeanPut])
def test_error_decreases_with_steps(payoff_cls):
    hedger = setup_hedger(payoff_cls)
    r = 0.02

    steps_low = 3
    steps_high = 200
    n_paths = 200

    hedger.engine.steps = steps_low
    _, err_low, *_ = hedger.run(r, n_paths)

    hedger.engine.steps = steps_high
    _, err_high, *_ = hedger.run(r, n_paths)

    std_low = np.std(err_low)
    std_high = np.std(err_high)

    assert std_high < std_low


@pytest.mark.parametrize("payoff_cls", [EuropeanCall, EuropeanPut])
def test_mean_error_close_to_zero(payoff_cls):
    hedger = setup_hedger(payoff_cls)
    r, steps = 0.02, 200
    n_paths = 500

    errors = []
    hedger.engine.steps = steps
    _, errors, *_ = hedger.run(r, n_paths)

    mean_error = np.mean(errors)

    assert abs(mean_error) < 1e-2


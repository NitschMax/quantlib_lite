import numpy as np
import pytest

from quantlib_lite.energy import DeliveryPeriod, ForwardTrade, MarketCurve
from quantlib_lite.energy import calendar as cld
from quantlib_lite.stochastic_models import SchwartzSmith

MODEL = SchwartzSmith(mu_zeta=0.0, sigma_zeta=0.3, theta=1.0, sigma_xi=0.3,
                      zeta_0=np.log(50), xi_0=0.0, rho=0.3, A=0.1, t_w=0.0)
STATE = MODEL.initial_state()


# --- calendar ---------------------------------------------------------------

def test_calendar_scalar_and_array():
    assert cld.days_in_month(3) == 30
    assert cld.hours_in_month(3) == 720
    assert np.array_equal(cld.days_in_month(np.arange(3)), [30, 30, 30])
    assert np.array_equal(cld.hours_in_month([0, 1]), [720, 720])
    assert cld.month_end(2) == cld.month_start(3)


def test_has_started_includes_boundary():
    assert cld.has_started(3, 0.25)
    assert not cld.has_started(4, 0.25)


# --- DeliveryPeriod ---------------------------------------------------------

def test_delivery_period():
    p = DeliveryPeriod(2, 3)
    assert list(p.months()) == [2, 3, 4]
    assert 4 in p and 5 not in p
    with pytest.raises(ValueError):
        DeliveryPeriod(0, 0)
    with pytest.raises(ValueError):
        DeliveryPeriod(-1)


# --- average_forward --------------------------------------------------------

def test_average_forward_uneven_counts_match_manual_mean():
    start = np.arange(3) / 12
    end = start + 1 / 12
    counts = np.array([10, 30, 7])
    result = MODEL.average_forward(0.0, start, end, STATE, counts)
    expected = [np.mean([MODEL.forward(0.0, s + (e - s) * k / n, STATE) for k in range(n)])
                for s, e, n in zip(start, end, counts)]
    assert result.shape == (3,)
    assert np.allclose(result, expected)


def test_average_forward_batch_state_and_bad_counts():
    start, end = np.arange(3) / 12, np.arange(1, 4) / 12
    states = np.stack([STATE, STATE * 1.1])
    assert MODEL.average_forward(0.0, start, end, states).shape == (2, 3)
    with pytest.raises(ValueError):
        MODEL.average_forward(0.0, start, end, STATE, 0)
    with pytest.raises(ValueError):
        MODEL.average_forward(0.0, start, end, STATE, 2.5)


# --- MarketCurve ------------------------------------------------------------

def test_market_curve_from_model_matches_average_forward():
    curve = MarketCurve.from_model(MODEL, 0.0, STATE, [0, 1, 2])
    expected = MODEL.average_forward(0.0, np.arange(3) / 12, np.arange(1, 4) / 12, STATE, 30)
    assert curve.months() == [0, 1, 2]
    assert np.allclose([curve.price(m) for m in curve.months()], expected)


def test_market_curve_validation():
    curve = MarketCurve(0.0, {0: 50.0})
    with pytest.raises(KeyError):
        curve.price(1)
    with pytest.raises(ValueError):
        MarketCurve.from_model(MODEL, 0.0, np.ones((2, 2)), [0])
    with pytest.raises(TypeError):
        MarketCurve.from_model(object(), 0.0, STATE, [0])


# --- ForwardTrade -----------------------------------------------------------

def test_forward_trade_value_and_started_months():
    curve = MarketCurve(0.0, {0: 60.0, 1: 70.0})
    trade = ForwardTrade(DeliveryPeriod(0, 2), volume=2.0, price=50.0)
    # month 0 starts at t=0, so only month 1 is open
    assert trade.open_months(curve) == [1]
    assert trade.value(curve) == pytest.approx(2.0 * 720 * (70.0 - 50.0))

    later = MarketCurve(0.0, {0: 60.0, 1: 70.0})
    both = ForwardTrade(DeliveryPeriod(1, 1), volume=-1.0, price=80.0)
    assert both.value(later) == pytest.approx(-1.0 * 720 * (70.0 - 80.0))


def test_forward_trade_validation():
    with pytest.raises(ValueError):
        ForwardTrade(DeliveryPeriod(0), volume=0.0, price=50.0)
    with pytest.raises(TypeError):
        ForwardTrade((0, 1), volume=1.0, price=50.0)
    with pytest.raises(TypeError):
        ForwardTrade(DeliveryPeriod(0), 1.0, 50.0).value({0: 1.0})

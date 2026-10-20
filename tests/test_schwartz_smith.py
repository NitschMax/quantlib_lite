import numpy as np
import pytest

from quantlib_lite.stochastic_models import SchwartzSmith


MODELS = [
    SchwartzSmith(mu_zeta=0.0, sigma_zeta=0.1, theta=1.0, sigma_xi=0.1, zeta_0=1.0, xi_0=0.0, rho=0.5, A=0.1, t_w=0.0),
    SchwartzSmith(mu_zeta=0.05, sigma_zeta=0.2, theta=3.0, sigma_xi=0.4, zeta_0=0.5, xi_0=0.3, rho=-0.4, A=0.2, t_w=0.25),
]


def analytic_vol_short_limit(model):
    """Forward vol for tau -> 0: both factors fully contribute."""
    return np.sqrt(model.sigma_zeta**2 + model.sigma_xi**2 + 2 * model.rho * model.sigma_zeta * model.sigma_xi)


def analytic_vol_long_limit(model):
    """Forward vol for tau -> inf: the mean-reverting factor dies out."""
    return model.sigma_zeta


# ---------------------------------------------------------------------------
# 1. Monte Carlo forward prices within 3 standard errors of the closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("T", [0.25, 1.0, 3.0])
def test_monte_carlo_forward_matches_analytic_forward(model, T):
    n_paths, steps = 200_000, 8
    rng = np.random.default_rng(1234)

    X_T = model.state_paths_batch(T, steps, n_paths, rng=rng)[:, -1]
    S_T = np.exp(X_T[:, 0] + X_T[:, 1] + model.seasonal_component(T))

    mc_forward = S_T.mean()
    std_error = S_T.std(ddof=1) / np.sqrt(n_paths)
    exact_forward = model.forward(0.0, T, model.initial_state())

    assert abs(mc_forward - exact_forward) < 3 * std_error


@pytest.mark.parametrize("model", MODELS)
def test_monte_carlo_forward_is_martingale_in_t(model):
    # E[F(t, T, X_t)] = F(0, T, X_0) for every t <= T
    n_paths, steps, T = 200_000, 8, 2.0
    rng = np.random.default_rng(4321)

    state = model.state_paths_batch(T, steps, n_paths, rng=rng)
    times = model.times(T, steps)
    forwards = model.forward(times, T, state)

    mc_forward = forwards.mean(axis=0)
    std_error = forwards.std(axis=0, ddof=1) / np.sqrt(n_paths)
    exact_forward = model.forward(0.0, T, model.initial_state())

    # at t = T the forward collapses onto the spot S_T
    assert forwards[:, -1] == pytest.approx(np.exp(state[:, -1].sum(axis=-1) + model.seasonal_component(T)))
    # at t = 0 there is no randomness, so test the remaining dates only
    assert np.all(np.abs(mc_forward[1:] - exact_forward) < 3 * std_error[1:])


# ---------------------------------------------------------------------------
# 2. Limits of the forward volatility for short and long delivery times
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_forward_vol_converges_to_short_delivery_limit(model):
    t = 0.5
    taus = np.logspace(-1, -8, 8)
    errors = np.abs(model.forward_vol(t, t + taus) - analytic_vol_short_limit(model))

    assert np.all(np.diff(errors) < 0)
    assert errors[-1] == pytest.approx(0.0, abs=1e-7)
    assert model.forward_vol(t, t) == pytest.approx(analytic_vol_short_limit(model), rel=1e-12)


@pytest.mark.parametrize("model", MODELS)
def test_forward_vol_converges_to_long_delivery_limit(model):
    t = 0.5
    taus = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
    errors = np.abs(model.forward_vol(t, t + taus) - analytic_vol_long_limit(model))

    # non-increasing: for fast mean reversion the error hits machine zero early
    assert np.all(np.diff(errors) <= 0)
    assert errors[0] > errors[-1]
    assert errors[-1] == pytest.approx(0.0, abs=1e-12)
    assert model.forward_vol(t, t + 1e4) == pytest.approx(analytic_vol_long_limit(model), rel=1e-12)


# ---------------------------------------------------------------------------
# 3. forward_vol^2 equals d/dtau of the total variance of log S_T
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tau", [1e-3, 0.1, 0.5, 1.0, 2.0, 10.0])
def test_forward_vol_squared_is_time_derivative_of_total_variance(model, tau):
    h = 1e-5
    total_variance = lambda dt: model.covariance_matrix(dt).sum(axis=(0, 1))
    numerical_derivative = (total_variance(tau + h) - total_variance(tau - h)) / (2 * h)

    t = 0.3
    assert model.forward_vol(t, t + tau)**2 == pytest.approx(numerical_derivative, rel=1e-8)


@pytest.mark.parametrize("model", MODELS)
def test_forward_vol_squared_is_time_derivative_of_total_variance_vectorised(model):
    h = 1e-5
    taus = np.linspace(0.01, 5.0, 50)
    total_variance = lambda dt: model.covariance_matrix(dt).sum(axis=(0, 1))
    numerical_derivative = (total_variance(taus + h) - total_variance(taus - h)) / (2 * h)

    assert model.forward_vol(0.0, taus)**2 == pytest.approx(numerical_derivative, rel=1e-8)


# ---------------------------------------------------------------------------
# 4. Broadcasting of forward / log_forward over every float/array combination
# ---------------------------------------------------------------------------

N_STATES, N_DELIVERIES = 3, 4
STATE_VECTOR = np.array([0.8, -0.2])
STATE_BATCH = np.array([[0.8, -0.2], [1.0, 0.0], [1.3, 0.4]])
T_GRID = np.array([1.0, 1.5, 2.0, 3.0])

# (t, T, state, expected output shape)
# Output convention: state.shape[:-1] + T.shape, with t broadcasting against it.
BROADCAST_CASES = {
    "t float, T float, state vector": (0.5, 2.0, STATE_VECTOR, ()),
    "t float, T array, state vector": (0.5, T_GRID, STATE_VECTOR, (N_DELIVERIES,)),
    "t array, T float, state vector": (np.array([0.0, 0.25, 0.5]), 2.0, STATE_VECTOR, (N_STATES,)),
    "t array, T array, state vector": (np.array([0.0, 0.25, 0.5, 0.75]), T_GRID, STATE_VECTOR, (N_DELIVERIES,)),
    "t float, T float, state batch": (0.5, 2.0, STATE_BATCH, (N_STATES,)),
    "t float, T array, state batch": (0.5, T_GRID, STATE_BATCH, (N_STATES, N_DELIVERIES)),
    "t array, T float, state batch": (np.array([0.0, 0.25, 0.5]), 2.0, STATE_BATCH, (N_STATES,)),
    "t array, T array, state batch": (np.array([[0.0], [0.25], [0.5]]), T_GRID, STATE_BATCH, (N_STATES, N_DELIVERIES)),
}


def scalar_reference(model, t, T, state, shape):
    """Evaluate the forward element by element with scalar inputs only."""
    state = np.asarray(state, float)
    t_b = np.broadcast_to(np.asarray(t, float), shape)
    T_b = np.broadcast_to(np.asarray(T, float), shape)
    state_b = np.broadcast_to(state.reshape(state.shape[:-1] + (1,) * np.ndim(T) + (2,)), shape + (2,))

    out = np.empty(shape)
    for idx in np.ndindex(shape):
        out[idx] = model.forward(float(t_b[idx]), float(T_b[idx]), state_b[idx])
    return out


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("t, T, state, expected_shape", BROADCAST_CASES.values(), ids=BROADCAST_CASES.keys())
def test_forward_broadcasts_over_t_T_and_state(model, t, T, state, expected_shape):
    forward = model.forward(t, T, state)

    assert np.shape(forward) == expected_shape
    assert forward == pytest.approx(scalar_reference(model, t, T, state, expected_shape), rel=1e-12)
    assert model.log_forward(t, T, state) == pytest.approx(np.log(forward), rel=1e-12)


@pytest.mark.parametrize("model", MODELS)
def test_forward_scalar_inputs_match_closed_form(model):
    t, T = 0.5, 2.0
    zeta, xi = STATE_VECTOR
    tau = T - t
    expected = np.exp(
        zeta + model.mu_zeta * tau
        + xi * np.exp(-model.theta * tau)
        + model.seasonal_component(T)
        + 0.5 * model.covariance_matrix(tau).sum()
    )

    assert model.forward(t, T, STATE_VECTOR) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("model", MODELS)
def test_forward_broadcasts_over_simulated_state_paths(model):
    # usage from examples/basic_usage_schwartz_smith.py: t is the path time grid
    T, steps, n_paths = 1.0, 12, 5
    state = model.state_paths_batch(T, steps, n_paths, rng=np.random.default_rng(0))
    times = model.times(T, steps)

    forward = model.forward(times, T, state)

    assert forward.shape == (n_paths, steps + 1)
    for i in range(n_paths):
        for k in range(steps + 1):
            assert forward[i, k] == pytest.approx(model.forward(times[k], T, state[i, k]), rel=1e-12)

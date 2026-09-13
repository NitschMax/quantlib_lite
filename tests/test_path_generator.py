import numpy as np
import pytest

from quantlib_lite.path.path import Path
from quantlib_lite.path_generator import PathGenerator, PythonPathGenerator
from quantlib_lite.stochastic_models import GBM, JumpDiffusion, OrnsteinUhlenbeck, StochasticModel


class SpyModel(StochasticModel):
    """Records the arguments it was called with and returns a fixed sentinel."""

    def __init__(self, sentinel):
        self._sentinel = sentinel
        self.calls = []

    @property
    def dimension(self):
        return 1

    def sample_paths_batch(self, T, steps, n_paths, rng=None):
        self.calls.append((T, steps, n_paths, rng))
        return self._sentinel


def test_python_path_generator_is_a_path_generator():
    generator = PythonPathGenerator()
    assert isinstance(generator, PathGenerator)


def test_generate_delegates_to_model_sample_paths_batch():
    sentinel = object()
    model = SpyModel(sentinel)
    generator = PythonPathGenerator()
    rng = np.random.default_rng(0)

    result = generator.generate(model, T=1.0, steps=10, n_paths=5, rng=rng)

    assert result is sentinel
    assert model.calls == [(1.0, 10, 5, rng)]


@pytest.mark.parametrize(
    "model",
    [
        GBM(mu=0.05, sigma=0.2),
        JumpDiffusion(mu=0.05, sigma=0.2, lam=1.0, jump_mean=-0.05, jump_std=0.1),
        OrnsteinUhlenbeck(mu=1.0, sigma=0.2, theta=0.5, X0=0.5),
    ],
)
def test_generate_returns_correct_number_and_shape_of_paths(model):
    T, steps, n_paths = 1.0, 10, 7
    generator = PythonPathGenerator()
    rng = np.random.default_rng(0)

    paths = generator.generate(model, T, steps, n_paths, rng)

    assert len(paths) == n_paths
    assert all(isinstance(p, Path) for p in paths)
    assert all(len(p) == steps + 1 for p in paths)


@pytest.mark.parametrize(
    "model, expected_x0",
    [
        (GBM(mu=0.05, sigma=0.2), 1.0),
        (JumpDiffusion(mu=0.05, sigma=0.2, lam=1.0, jump_mean=-0.05, jump_std=0.1), 1.0),
        (OrnsteinUhlenbeck(mu=1.0, sigma=0.2, theta=0.5, X0=0.5), 0.5),
    ],
)
def test_generated_paths_start_at_model_initial_value(model, expected_x0):
    generator = PythonPathGenerator()
    rng = np.random.default_rng(0)

    paths = generator.generate(model, T=1.0, steps=10, n_paths=4, rng=rng)

    assert all(p.values[0] == pytest.approx(expected_x0) for p in paths)


def test_generate_shares_times_with_model():
    model = GBM(mu=0.05, sigma=0.2)
    generator = PythonPathGenerator()
    rng = np.random.default_rng(0)
    T, steps = 1.0, 10

    paths = generator.generate(model, T, steps, n_paths=3, rng=rng)

    expected_times = tuple(model.times(T, steps))
    assert all(p.times == expected_times for p in paths)


def test_generate_is_reproducible_with_same_seed():
    model = GBM(mu=0.05, sigma=0.2)
    generator = PythonPathGenerator()
    T, steps, n_paths = 1.0, 10, 5

    paths_a = generator.generate(model, T, steps, n_paths, rng=np.random.default_rng(42))
    paths_b = generator.generate(model, T, steps, n_paths, rng=np.random.default_rng(42))

    for pa, pb in zip(paths_a, paths_b):
        assert pa.values == pytest.approx(pb.values)


def test_generate_produces_independent_paths():
    model = GBM(mu=0.05, sigma=0.2)
    generator = PythonPathGenerator()
    rng = np.random.default_rng(0)

    paths = generator.generate(model, T=1.0, steps=10, n_paths=5, rng=rng)

    distinct_final_values = {round(p.values[-1], 12) for p in paths}
    assert len(distinct_final_values) == len(paths)

from quantlib_lite.path.path import Path
from .stochastic_model import StochasticModel
import numpy as np

class GBM(StochasticModel):
    def __init__(self, mu, sigma, S0=1.0):
        self.__mu = float(mu)
        self.__sigma = float(sigma)
        self.__S0 = float(S0)

    @property
    def mu(self):
        return self.__mu

    @property
    def sigma(self):
        return self.__sigma

    @property
    def S0(self):
        return self.__S0

    @property
    def dimension(self):
        return 1

    def __hash__(self):
        return hash((self.mu, self.sigma, self.S0))

    def __eq__(self, other):
        if isinstance(other, GBM):
            return (self.mu, self.sigma, self.S0) == (other.mu, other.sigma, other.S0)
        else:
            return NotImplemented

    def sample_paths_batch(self, T, steps, n_paths, rng=None):
        times = self.times(T, steps)
        dt = self.dt(T, steps)

        if rng == None:
            rng = np.random.default_rng()

        dW = rng.normal(0, np.sqrt(dt), (n_paths, steps))

        W = np.cumsum(dW, axis=1)
        W = np.insert(W, 0, 0, axis=1)
        X = self.S0 * np.exp((self.mu - 0.5 * self.sigma ** 2) * times[None, :] + self.sigma * W)
        return [Path(times, x) for x in X]

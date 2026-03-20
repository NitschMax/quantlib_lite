from path.path import Path
from .stochastic_model import StochasticModel
import numpy as np

class GBM(StochasticModel):
    def __init__(self, mu, sigma):
        self.__mu = float(mu)
        self.__sigma = float(sigma)

    @property
    def mu(self):
        return self.__mu

    @property
    def sigma(self):
        return self.__sigma

    @property
    def dimension(self):
        return 1

    def sample_path(self, T, steps):
        times = self.times(T, steps)
        dt = self.dt(T, steps)
        dW = np.random.normal(0, np.sqrt(dt), steps)
        W = np.cumsum(dW)
        W = np.insert(W, 0, 0)
        X = np.exp((self.mu - 0.5 * self.sigma ** 2) * times + self.sigma * W)
        return Path(times, X)

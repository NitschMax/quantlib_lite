import numpy as np

from quantlib_lite.path.path import Path
from .stochastic_model import StochasticModel

class OrnsteinUhlenbeck(StochasticModel):
    def __init__(self, mu, sigma, theta, X0=1):
        self.__mu = float(mu)       # long-term mean
        self.__sigma = float(sigma) # volatitlity
        self.__theta = float(theta) # mean reversion speed
        self.__X0 = float(X0)       # initial value

    @property
    def mu(self):
        return self.__mu

    @property
    def sigma(self):
        return self.__sigma

    @property
    def theta(self):
        return self.__theta

    @property
    def X0(self):
        return self.__X0

    @property
    def dimension(self):
        return 1

    def __hash__(self):
        return hash((self.mu, self.sigma, self.theta, self.X0))
    
    def __eq__(self, other):
        if isinstance(other, OrnsteinUhlenbeck):
            return (self.mu, self.sigma, self.theta, self.X0) == (other.mu, other.sigma, other.theta, other.X0)
        else:
            return NotImplemented

    def sample_path(self, T, steps):
        times = self.times(T, steps)
        dt = self.dt(T, steps)

        X = [self.X0]
        for k in range(steps):
            X_new = X[k] + self.theta * (self.mu - X[k]) * dt + self.sigma * np.sqrt(dt) * np.random.normal(0, 1)
            X.append(X_new)
        return Path(times, X)


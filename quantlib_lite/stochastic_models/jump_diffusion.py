from quantlib_lite.path.path import Path
from .stochastic_model import StochasticModel
import numpy as np

class JumpDiffusion(StochasticModel):
    def __init__(self, mu, sigma, lam, jump_mean, jump_std):
        self.__mu = float(mu)                   # drift term
        self.__sigma = float(sigma)             # volatility
        self.__lam = float(lam)           # jump frequency
        self.__jump_mean = float(jump_mean)     # mean of log jump size
        self.__jump_std = float(jump_std)       # std of jump size

    @property
    def mu(self):
        return self.__mu

    @property
    def sigma(self):
        return self.__sigma

    @property
    def lam(self):
        return self.__lam

    @property
    def jump_mean(self):
        return self.__jump_mean

    @property
    def jump_std(self):
        return self.__jump_std

    @property
    def dimension(self):
        return 3

    def __hash__(self):
        return hash((self.mu, self.sigma))

    def __eq__(self, other):
        if isinstance(other, GBM):
            return (self.mu, self.sigma) == (other.mu, other.sigma)
        else:
            return NotImplemented

    def sample_path(self, T, steps):
        times = self.times(T, steps)
        dt = self.dt(T, steps)
        rng = np.random.default_rng()

        dW = rng.normal(0, np.sqrt(dt), steps)
        W = np.cumsum(dW)
        W = np.insert(W, 0, 0)

        # Simulate jump times and sizes
        dN = rng.poisson(self.lam * dt, steps)
        dN = np.insert(dN, 0, 0)  # Insert initial value for cumulative sum
        print(dN)

        # Calculate if jumps occurred at each time step
        jump_occurred = dN > 0 
        print(jump_occurred)
        jump_component = np.zeros_like(times)
        jump_component[jump_occurred] = rng.normal(self.jump_mean * dN[jump_occurred], self.jump_std * np.sqrt(dN[jump_occurred]))
        jumps_accumulated = np.cumsum(jump_component)
        print(jumps_accumulated)

        k = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1
        # Modified GBM
        X = np.exp((self.mu - 0.5 * self.sigma ** 2 - k * self.lam) * times + self.sigma * W) * np.exp(jumps_accumulated)

        return Path(times, X)


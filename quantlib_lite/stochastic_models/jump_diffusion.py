from quantlib_lite.path.path import Path
from .stochastic_model import StochasticModel
import numpy as np

class JumpDiffusion(StochasticModel):
    def __init__(self, mu, sigma, lam, jump_mean, jump_std, S0=1.0):
        self.__mu = float(mu)                   # drift term
        self.__sigma = float(sigma)             # volatility
        self.__lam = float(lam)           # jump frequency
        self.__jump_mean = float(jump_mean)     # mean of log jump size
        self.__jump_std = float(jump_std)       # std of jump size
        self.__S0 = float(S0)                   # initial value

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
    def S0(self):
        return self.__S0

    @property
    def dimension(self):
        return 3

    def __hash__(self):
        return hash((self.mu, self.sigma, self.lam, self.jump_mean, self.jump_std, self.S0))

    def __eq__(self, other):
        if isinstance(other, JumpDiffusion):
            return (self.mu, self.sigma, self.lam, self.jump_mean, self.jump_std, self.S0) == (other.mu, other.sigma, other.lam, other.jump_mean, other.jump_std, other.S0)
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

        # Simulate jump times and sizes
        dN = rng.poisson(self.lam * dt, (n_paths, steps))  # Number of jumps in each time step
        dN = np.insert(dN, 0, 0, axis=1)  # Insert initial value for cumulative sum

        # Calculate if jumps occurred at each time step
        jump_occurred = dN > 0 
        jump_component = np.zeros((n_paths, len(times)))
        jump_component[jump_occurred] = rng.normal(self.jump_mean * dN[jump_occurred], self.jump_std * np.sqrt(dN[jump_occurred]))
        jumps_accumulated = np.cumsum(jump_component, axis=1)

        k = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1
        # Modified GBM
        X = self.S0 * np.exp((self.mu - 0.5 * self.sigma ** 2 - k * self.lam) * times[None, :] + self.sigma * W) * np.exp(jumps_accumulated)

        return [Path(times, x) for x in X]


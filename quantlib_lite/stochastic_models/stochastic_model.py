from abc import ABC, abstractmethod
import numpy as np

class StochasticModel(ABC):

    @property
    @abstractmethod
    def dimension(self):
        """Declare number of stochastic factors."""

    @abstractmethod
    def sample_path(self, T, steps):
        """"Provide a single sample of a random path as defined by the stochastic model

        T      maturity
        steps  number of timesteps

        return a Path object Path(self.times(T, steps), values) of the sampled path
        """
    
    def times(self, T, steps):
        return np.linspace(0, T, steps+1)

    def dt(self, T, steps):
        return float(T/steps)



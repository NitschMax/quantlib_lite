from abc import ABC, abstractmethod
import numpy as np

class StochasticModel(ABC):

    @property
    @abstractmethod
    def dimension(self):
        """Declare number of stochastic factors."""

    @abstractmethod
    def sample_paths_batch(self, T, steps, n_paths, rng=None):
        """"Provide a single sample of a random path as defined by the stochastic model

        T      maturity
        steps  number of timesteps
        rng     optional random number generator to ensure reproducability

        return an array of Path objects Path(self.times(T, steps), values) of the sampled paths
        """
    
    def times(self, T, steps):
        return np.linspace(0, T, steps+1)

    def dt(self, T, steps):
        return float(T/steps)



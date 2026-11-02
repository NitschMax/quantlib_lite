import numpy as np
from abc import ABC
from abc import abstractmethod
from quantlib_lite.path.path import Path

class StochasticModel(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Declare number of stochastic factors."""

    @abstractmethod
    def sample_paths_batch(self, T: float, steps: int, n_paths: int, rng=None) -> list[Path]:
        """"Provide a batch sample of a random paths as defined by the stochastic model

        T      maturity
        steps  number of timesteps
        rng     optional random number generator to ensure reproducability

        return an array of Path objects Path(self.times(T, steps), values) of the sampled paths
        """
    
    def times(self, T, steps):
        return np.linspace(0, T, steps+1)

    def dt(self, T, steps):
        return float(T/steps)



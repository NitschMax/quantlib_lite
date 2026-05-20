import numpy as np
from quantlib_lite.stochastic_models import StochasticModel

class SimulationEngine:
    def __init__(self, model, T, steps, seed=1):
        if isinstance(model, StochasticModel):
            self._model = model
        else:
            raise TypeError('model must be an instance of StochasticModel')

        self.T = float(T)
        self.steps = int(steps)

        self.seed = int(seed)
        self._cache = {}

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if isinstance(value, StochasticModel):
            self._model = value
        else:
            raise TypeError('model must be an instance of StochasticModel')

    def simulation_key(self):
        return (self.model, self.T, self.steps, self.seed)

    def simulate(self, n_paths):
        key = self.simulation_key()

        if key in self._cache:
            paths, rng = self._cache[key]
        else:
            rng = np.random.default_rng(self.seed)
            paths = []

        len_diff = int(n_paths) - len(paths)
        if len_diff > 0:
            paths.extend([self.model.sample_path(self.T, self.steps, rng=rng) for _ in range(len_diff)])
            self._cache[key] = (paths, rng)

        return np.array(paths[:n_paths])


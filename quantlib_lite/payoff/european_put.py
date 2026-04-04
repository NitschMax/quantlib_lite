import numpy as np
from .payoff import Payoff

class EuropeanPut(Payoff):
    def __init__(self, K):
        self.__K = float(K)

    @property
    def K(self):
        return self.__K

    def evaluate(self, path):
        return np.maximum(self.K - self.terminal_value(path), 0.0)

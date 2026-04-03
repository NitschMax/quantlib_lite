import numpy as np
from .payoff import Payoff

class EuropeanCall(Payoff):
    def __init__(self, K):
        self.__K = float(K) # Strike price K

    @property
    def K(self):
        return self.__K

    def evaluate(self, path):
        return np.maximum(self.terminal_value(path) - self.K, 0.0)

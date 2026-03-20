from .payoff import Payoff
import numpy as np

class AsianCall(Payoff):
    def __init__(self, K):
        self.__K = float(K) # strike price K

    @property
    def K(self):
        return self.__K

    def evaluate(self, path):
        return np.maximum( np.mean(path.values) - self.K, 0.0 )


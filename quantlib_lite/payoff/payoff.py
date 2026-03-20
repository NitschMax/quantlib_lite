from abc import ABC, abstractmethod

class Payoff(ABC):
    @abstractmethod
    def evaluate(self, path):
        """Calculate the payout for the given payoff class based on the given path

        path    Path object which contains a time grid + the simulation values

        return the payout as a single floating number
        """

    def terminal_value(self, path):
        return path.values[-1]


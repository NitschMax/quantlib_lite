from abc import ABC, abstractmethod

class RiskMeasure(ABC):

    @abstractmethod
    def evaluate(self, values):
        """"Takes in an array of price values and returns a riskadjusted price

        values  iterable of floats

        return  single floating number resembling the risk adjusted price
        """


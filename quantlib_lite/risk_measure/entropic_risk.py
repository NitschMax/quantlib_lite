from .risk_measure import RiskMeasure
import numpy as np

class EntropicRisk(RiskMeasure):
    def __init__(self, theta):
        self.__theta = float(theta)
    
    @property
    def theta(self):
        return self.__theta

    def evaluate(self, values):
        return np.log(np.mean(np.exp(self.theta * np.array(values)) ) ) / self.theta


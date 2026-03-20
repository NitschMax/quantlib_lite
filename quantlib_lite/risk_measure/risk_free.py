from .risk_measure import RiskMeasure
import numpy as np

class RiskFree(RiskMeasure):
    def evaluate(self, values):
        return np.mean(np.array(values))


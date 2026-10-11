from abc import ABC, abstractmethod
import numpy as np

class ForwardCurveModel(ABC):
    """A model with a closed-form forward curve, given a model state."""

    @abstractmethod
    def forward(self, t, T, state):
        """"Provide a single sample of a random path as defined by the stochastic model

        t      current time
        T      maturity
        state  current model state

        return a float of the forward price at time t for maturity T
        """

    @abstractmethod
    def forward_vol(self, t, T):
        """""Return the forward volatility
        """

    @abstractmethod
    def initial_state(self):
        """""Return the initial state of the model
        """

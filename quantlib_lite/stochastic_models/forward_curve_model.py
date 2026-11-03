from abc import ABC, abstractmethod
import numpy as np

class ForwardCurveModel(ABC):
    """A model with a closed-form forward curve, given a model state."""

    @abstractmethod
    def forward(self, t, T, state):
        """Calculate the forward price from time t to maturity T for state

        t      current time
        T      maturity
        state  current model state

        return a float of the forward price at time t for maturity T
        """

    @abstractmethod
    def forward_vol(self, t, T):
        """Return the forward volatility
        """

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Return the initial state of the model
        """

    def monthly_forward(self, t, months, state, points_per_month=30):
        """Calculate the average forward value of the model on the given day.
            For simplicity we assume 360 days per year and 30 days per month.
        
        t       current date as scalar marking the current time
        months  an iterable of ints
        state   state vector at current time

        returns the average price of each month for each given state
        """

        # Check if t * 12 > month satisfied for all months with array broadcasting
        if (t * 12 > months).any():
            raise ValueError("The current date is after the requested month.")

        months = np.asarray(months, int)
        months = months.reshape(months.shape + (1,))

        # rewrite into broadcasted array
        forward_values = np.mean(self.forward(t, (np.linspace(0, 1, points_per_month, endpoint=False) + months)/12, state), axis=-1)

        return np.asarray(forward_values)


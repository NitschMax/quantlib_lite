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
        
        t       current dat as scalar marking the time in the year
        months  an iterable of ints
        state   state vector at current time

        returns the average price of each month for each given state
        """

        forward_values = []
        for month in months:
            if t * 12 > month:
                raise ValueError("The current date is after the requested month.")
            else:
                T_arr = np.linspace(month/12.0, (month+1)/12.0, points_per_month, endpoint=False)
                forward_values.append(np.mean(self.forward(t, T_arr, state), axis=-1))

        return np.asarray(forward_values)


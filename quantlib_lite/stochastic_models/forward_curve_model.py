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

    def average_forward(self, t, start, end, state, points_per_period=30):
        """Calculate the average forward value between start and end of the model on the given day.
        
        t       current date as scalar marking the current time
        start   start time of periond in years
        end      end time of periond in years
        points_per_period the number of equidistant samples per period; an integer or an integer
                  array broadcastable to the shape of start (one count per period)
        state   state vector at current time

        returns the average price of the period each given state
        """

        start = np.asarray(start, float)
        end = np.asarray(end, float)

        # Compare array shapes and make sure that elementwise start < end
        if not start.shape == end.shape:
            raise ValueError("arrays must have equal dimensions")
        if np.any(start >= end):
            raise ValueError("start date must be strictly less than end date")

        # Check that time is before all start dates
        if start.min() + 1e-12 < t: raise ValueError("time must be strictly smaller than start date")

        n = np.asarray(points_per_period)
        if not np.issubdtype(n.dtype, np.integer) or np.any(n < 1):
            raise ValueError("points_per_period must be positive integers")
        n = np.broadcast_to(n, start.shape)[..., None]

        # Sample k/n of the way through each period for k < n. Periods with fewer
        # samples than the maximum are padded and masked out of the mean.
        k = np.arange(n.max())
        grid = start[..., None] + (end - start)[..., None] * (k / n)
        values = self.forward(t, grid, state)
        forward_values = np.sum(np.where(k < n, values, 0.0), axis=-1) / n[..., 0]

        return np.asarray(forward_values)


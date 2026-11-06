from dataclasses import dataclass
from . import calender as cld

@dataclass(frozen=True)
class DeliveryPeriod:
    """
    DeliveryPeriod: a class that represents a delivery period of time.
    Attributes: first_month that cover the time range [m/12, (m+1)/12] from the start of the month, n_months (number of months following)
    """
    first_month: int
    n_months: int = 1

    def __post_init__(self):
        if self.n_months < 1: raise ValueError('n_months must be greater or equal than 1.')
        if self.first_month < 0: raise ValueError('first_month must be greater or equal than 0.')

    def months(self) -> range:
        return range(self.first_month, self.n_months + self.first_month)

    def start_time(self) -> float:
        return cld.month_start(self.first_month)

    def __contains__(self, month)  -> bool:
        return month in self.months()

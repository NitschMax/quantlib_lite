from dataclasses import dataclass

@dataclass(frozen=True)
class DeliveryPeriod:
    """
    DeliveryPeriod: a class that represents a delivery period of time.
    Attributes: first_month, n_months (number of months)
    """
    first_month: int
    n_months: int = 1

    def __post_init__(self):
        if self.n_months < 1: raise Exception('n_months must be greater or equal than 0.')
        if self.first_month < 0: raise Exception('first_month must be greater or equal than 0.')

    def months(self) -> range:
        return range(self.first_month, self.n_months + self.first_month)

    def start_time(self) -> float:
        return self.first_month / 12

    def __contains__(self, month)  -> bool:
        return month in self.months()

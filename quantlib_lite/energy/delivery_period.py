from dataclasses import dataclass

@dataclass(frozen=True)
class DeliveryPeriod:
    first_month: int
    n_months: int = 1

    def months(self):
        return range(self.first_month, self.n_months + self.first_month)

    def start_time(self):
        return self.first_month / 12

    def __contains__(self, month):
        return month in self.months()

from dataclasses import dataclass

from .delivery_period import DeliveryPeriod
from .market_curve import MarketCurve
from . import calendar as cld


@dataclass(frozen=True)
class ForwardTrade:
    """
    Forward Trade dataclass.
    """
    period: DeliveryPeriod
    volume: float           # in MW positive for a buy
    price: float            # in Eur/MWh
    trade_id: str = ''

    def __post_init__(self):
        if not isinstance(self.period, DeliveryPeriod):
            raise TypeError(
                f"Argument period: expected dataclass of type DeliveryPeriod,"
                f" received {self.period.__class__.__name__}"
            )
        if self.volume == 0:
            raise ValueError('Expected volume different from zero')

    def open_months(self, curve):
        if not isinstance(curve, MarketCurve): raise TypeError('Please provide a MarketCurve object')

        # To obtain the valid months filter out all monts that have already started; the bool cld.has_started(month, curve.t) decides that
        open_months = [month for month in self.period.months() if not cld.has_started(month, curve.t)]
        return list(open_months)

    def value(self, curve):
        open_months = self.open_months(curve)
        value = self.volume * sum([cld.hours_in_month(month) * (curve.price(month) - self.price) for month in open_months])
        return value

    def cash_flow(self, month, settlement_price):
        if month not in self.period:
            raise ValueError('Cannot compute the cash flow if this month is not part of the trades delivery period')
        return self.volume * cld.hours_in_month(month) * (settlement_price - self.price)


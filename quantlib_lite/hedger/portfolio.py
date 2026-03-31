class Portfolio():
    def __init__(self, cash_value, asset_count):
        self.cash_value = float(cash_value)
        self.asset_count = float(asset_count)

    def value_at_price_S(self, asset_price):
        return self.cash_value + self.asset_count * float(asset_price)

    def update(self, new_asset, asset_price):
        asset_change =  float(new_asset) - self.asset_count
        self.asset_count += asset_change
        self.cash_value -= float(asset_change) * float(asset_price)


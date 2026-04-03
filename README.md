# quantlib_lite

A lightweight Python library for quantitative modeling and Monte Carlo pricing of stochastic processes.

---

## Overview

`quantlib_lite` provides a modular framework to:

- define stochastic models
- simulate sample paths
- evaluate payoffs
- aggregate results via risk measures
- estimate prices using Monte Carlo methods
- implement a delta hedge for european call option within the Black Scholes framework

The design separates concerns cleanly.
For the Pricer it follows the design:

```
Model → Path → Payoff → RiskMeasure → Pricer
```

For the Hedger it follows the design:

```
Model → Path → Payoff → DeltaHedgingStrategy → Hedger
```
---

## Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Example

```python
from quantlib_lite.stochastic_models.gbm import GBM
from quantlib_lite.payoff.european_call import EuropeanCall
from quantlib_lite.risk_measures.risk_free import RiskFree
from quantlib_lite.pricer import Pricer
from quantlib_lite.hedger import Hedger

model = GBM(mu=0.05, sigma=0.2)
payoff = EuropeanCall(K=1.0)
risk = RiskFree()

pricer = Pricer(model, payoff, risk)

price = pricer.price(T=1.0, steps=100, samples=1000)

print(price)
T = 1.0
r = 0.02
n_paths = 1000
steps = 100
strategy = DeltaHedgingStrategy()

hedger = Hedger(model, payoff, strategy)
for _ in range(n_paths):
    pf, error, S_T, payout = hedger.run(T, r, steps)
```

---

## Project Structure

```
quantlib_lite/
├── stochastic_models/   # stochastic processes (e.g. GBM, OU)
├── payoff/              # payoff definitions (e.g. European, Asian)
├── risk_measures/       # aggregation (mean, entropic risk)
├── pricer/              # Monte Carlo pricing logic
├── hedger/              # delta hedging logic
├── tests/               # unit tests
```

---

## Testing

Run tests with:

```bash
pytest
```

Tests are also executed automatically via GitHub Actions on each push.

---

## Notes

- The library is intentionally minimal and focused on clarity
- Designed for learning, experimentation, and extension
- Easily extendable with new models, payoffs, and risk measures

---

## License

MIT

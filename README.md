# quantlib_lit

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

The library separates stochastic simulation infrastructure from financial evaluation logic.
The simulation logic is encapsulated in the 

```
Model → SimulationEngine → PathGenerator → Path

```
wokflow, while the financial evaluation logic is encapsulated in the Pricer and Hedger workflows.

The `SimulationEngine` delegates the actual path sampling to a `PathGenerator`, which is a thin
wrapper around a model's vectorized `sample_paths_batch`. Today this defaults to
`PythonPathGenerator`, a pure NumPy implementation; the abstraction leaves room for an
accelerated (e.g. C++) backend to be swapped in later for selected models without changing
`SimulationEngine` or any downstream `Pricer`/`Hedger` code.
For the Pricer it follows the design:

```
SimulationEngine → Path → Payoff → RiskMeasure → Pricer
```

For the Hedger it follows the design:

```
SimulationEngine → Path → Payoff → DeltaHedgingStrategy → Hedger
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
from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall
from quantlib_lite.risk_measure import RiskFree
from quantlib_lite.hedger import DeltaHedgingStrategy, Hedger
from quantlib_lite.simulation_engine import SimulationEngine
from quantlib_lite import Pricer

seed = 42
model = GBM(mu=0.05, sigma=0.2, S0=1.0)
payoff = EuropeanCall(K=1.0)
risk = RiskFree()
engine = SimulationEngine(model, T=1.0, steps=100, seed=seed)

pricer = Pricer(engine, payoff, risk)

price = pricer.price(samples=1000)

print(price)

r = 0.02
n_paths = 1000
strategy = DeltaHedgingStrategy()

hedger = Hedger(engine, payoff, strategy)
pfs, errors, S_T_array, payouts = hedger.run(r, n_paths)
```

---

## Project Structure

```
quantlib_lite/
├── __init__.py
├── stochastic_models/   # stochastic processes (e.g. GBM, OU, JumpDiffusion)
├── simulation_engine/   # core simulation logic (caches and drives sampling)
├── path_generator/      # PathGenerator abstraction (PythonPathGenerator today,
│                        # pluggable accelerated backends later)
├── path/                # path representation
├── payoff/              # payoff definitions (e.g. European, Asian)
├── risk_measure/        # aggregation (mean, entropic risk)
├── pricer/              # Monte Carlo pricing logic
├── hedger/              # delta hedging logic
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

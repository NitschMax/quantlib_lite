import numpy as np
import matplotlib.pyplot as plt
from quantlib_lite.stochastic_models import JumpDiffusion
from quantlib_lite.payoff import EuropeanCall, EuropeanPut
from quantlib_lite.hedger import DeltaHedgingStrategy, Hedger
from quantlib_lite.path import Path
from quantlib_lite.simulation_engine import SimulationEngine


T = 1.0
r = 0.02
mu = 0.1
sigma = 0.2
K = 1.2

jump_mean = -0.1
jump_std = 0.3

steps = 100
n_paths = 1000
seed = 1

errors_mean = []
errors_std = []
portfolios_dict = {}
S_T_dict = {}
payouts_dict = {}

strategy = DeltaHedgingStrategy()

payoff = EuropeanPut(K=K)
payoff = EuropeanCall(K=K)
lam = 1.0
model = JumpDiffusion(mu, sigma, lam, jump_mean, jump_std)
engine = SimulationEngine(model, T, steps, seed=seed)
hedger = Hedger(engine, payoff, strategy)

lams = [1e-3, 1e-2, 1e-1, 1.0]
for lam in lams:
    hedger.engine.model = JumpDiffusion(mu, sigma, lam, jump_mean, jump_std)

    portfolios, errors, S_T_arr, payouts = hedger.run(r, n_paths)

    errors = np.array(errors)
    errors_mean.append(np.mean(errors))
    errors_std.append(np.std(errors))

    portfolios_dict[lam] = portfolios
    S_T_dict[lam] = S_T_arr
    payouts_dict[lam] = payouts

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, lam in enumerate(lams):
    ax = axes[i // 2, i % 2]
    ax.scatter(S_T_dict[lam], portfolios_dict[lam], alpha=0.5, label=f'λ={lam}')
    ordered_keys = np.argsort(S_T_dict[lam])
    ax.plot(np.array(S_T_dict[lam])[ordered_keys], np.array(payouts_dict[lam])[ordered_keys], color='k', label='Payoff')
    ax.set_title(f'Jump Diffusion with λ={lam}')
    ax.set_xlabel('S(T)')
    ax.set_ylabel('Portfolio Value at T')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()


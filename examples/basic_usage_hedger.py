import numpy as np
import matplotlib.pyplot as plt
from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall
from quantlib_lite.hedger import DeltaHedgingStrategy, Hedger
from quantlib_lite.path import Path


T = 1.0
r = 0.02
mu = 0.1
S0 = 1.0
sigma = 0.2
K = 1.0

steps_list = [3, 10, 30, 100]
steps_list = [3, 10, 30, 100, 300, 1000]
n_paths = 1000

errors_mean = []
errors_std = []
portfolios_dict = {}
S_T_dict = {}
payouts_dict = {}

strategy = DeltaHedgingStrategy()

model = GBM(mu=mu, sigma=sigma)
payoff = EuropeanCall(K=K)
hedger = Hedger(model, payoff, strategy)


for steps in steps_list:
    errors = []
    portfolios = []
    S_T_arr = []
    payouts = []

    for _ in range(n_paths):
        pf, error, S_T, payout = hedger.run(T, r, steps)

        portfolios.append(pf)
        errors.append(error)
        S_T_arr.append(S_T)
        payouts.append(payout)

    errors = np.array(errors)
    errors_mean.append(np.mean(errors))
    errors_std.append(np.std(errors))

    portfolios_dict[steps] = portfolios
    S_T_dict[steps] = S_T_arr
    payouts_dict[steps] = payouts


# Create analytical solution for comparison via inserting S_T into the payoff function

fig, axes = plt.subplots(np.floor_divide(len(steps_list), 2)+np.mod(len(steps_list), 2), 2, figsize=(10, 8))
axes = axes.transpose().flatten()
for idx, steps in enumerate(steps_list):
    ax = axes[idx]

    #Plot analytical result of payoff function
    S_T_arr = np.array(S_T_dict[steps])
    payoffs_arr = np.array([hedger.payoff.evaluate(Path([1], [S_T])) for S_T in S_T_arr])
    ordered_keys = np.argsort(S_T_arr)
    ax.plot(S_T_arr[ordered_keys], payoffs_arr[ordered_keys], color='k', lw=1)

    #Plot value of discretely hedged portfolio
    ax.grid(lw=0.5)
    ax.scatter(S_T_dict[steps], portfolios_dict[steps], alpha=0.1, label='Hedged with ' + str(steps) + ' steps')
    ax.set_xlabel('S_T')
    ax.set_ylabel('Payoff')
    ax.legend()

fig.suptitle('Replication of european call option via Black Scholes delta hedge')
fig.tight_layout()

plt.show()
plt.clf()
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(10, 8) )
for idx, err in enumerate([errors_mean, errors_std]):
    if idx == 0:
        err = np.abs(err)
    ax[idx].plot(steps_list, err, marker='o')
    ax[idx].set_xscale('log')
    ax[idx].set_xlabel('Steps')
    ax[idx].set_ylabel('Error')
    ax[idx].grid()
    ax[idx].set_title('Mean error' if idx == 0 else 'Std of error')

fig.suptitle('Error of delta hedging strategy')
fig.tight_layout()
plt.show()

plt.clf()
plt.close()

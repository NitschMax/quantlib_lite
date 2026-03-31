import numpy as np
import matplotlib.pyplot as plt
from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall
from quantlib_lite.hedger import DeltaHedgingStrategy, Hedger


T = 1.0
r = 0.1
S0 = 1
sigma = 0.2
K = 1.1

steps_list = [10, 20, 50, 100]
steps_list = [10, 20, 50, 100, 200, 500, 1000]
n_paths = 1000

errors_mean = []
errors_std = []

strategy = DeltaHedgingStrategy()

for steps in steps_list:
    errors = []

    for _ in range(n_paths):
        model = GBM(mu=r, sigma=sigma)
        payoff = EuropeanCall(K=K)

        hedger = Hedger(model, payoff, strategy)
        _, error = hedger.run(T, r, steps)

        errors.append(error)

    errors = np.array(errors)
    errors_mean.append(np.mean(errors))
    errors_std.append(np.std(errors))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(steps_list, np.abs(errors_mean), marker='o', label='Mean Error')
ax2.plot(steps_list, errors_std, marker='o', label='Standard Deviation of Error', color='orange')

plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Delta Hedging Error vs Time Discretization')
plt.legend()
plt.show()

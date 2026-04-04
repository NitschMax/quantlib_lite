from quantlib_lite.stochastic_models import StochasticModel
from quantlib_lite.payoff import EuropeanCall, EuropeanPut
import numpy as np
from scipy.stats import norm

class DeltaHedgingStrategy():
    def compute_a_t(self, t, S_t, T, r, sigma, K, payoff):
        tau = T - t

        if tau < 1e-14:
            if S_t > K:
                return 1
            else:
                return 0
            
        d1 = self.compute_d1(S_t, tau, r, sigma, K)

        # Caclulate the rebalanced portfolio for a european call option
        a_t = norm.cdf(d1)
        if isinstance(payoff, EuropeanCall):
            return a_t
        elif isinstance(payoff, EuropeanPut):
            return a_t - 1
        else:
            raise NotImplemented('Hedging Strategy only implemented for European Call and Put options')


    def compute_a_t_put(self, t, S_t, T, r, sigma, K):
        return self.compute_a_t(t, S_t, T, r, sigma, K) - 1

    def compute_initial_value(self, S_0, T, r, sigma, K, payoff):
        tau = T
        d1 = self.compute_d1(S_0, tau, r, sigma, K)
        d2 = self.compute_d2(S_0, tau, r, sigma, K)

        value_call = S_0 * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)

        if isinstance(payoff, EuropeanCall):
            return value_call
        elif isinstance(payoff, EuropeanPut):
            return value_call - S_0 + K * np.exp(-r*T)
        else:
            raise NotImplemented('Hedging Strategy only implemented for European Call and Put options')
    
    def compute_d1(self, S_t, tau, r, sigma, K):
        return (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / (sigma * tau**0.5)

    def compute_d2(self, S_t, tau, r, sigma, K):
        return self.compute_d1(S_t, tau, r, sigma, K) - sigma * tau**0.5


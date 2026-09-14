from quantlib_lite.stochastic_models import GBM, JumpDiffusion, OrnsteinUhlenbeck
from quantlib_lite.payoff import EuropeanCall
from quantlib_lite.risk_measure import RiskFree
from quantlib_lite.simulation_engine import SimulationEngine
from quantlib_lite import Pricer
from quantlib_lite.path_generator import CppPathGenerator, PythonPathGenerator

import time

def main():
    seed = 1
    mu = 0.05
    sigma = 0.2
    theta = 0.1
    X0 = 1.5
    lam = 1
    jump_mean = 0.1
    jump_std = 0.2

    model = GBM(mu, sigma)
    model = JumpDiffusion(mu, sigma, lam, jump_mean, jump_std)
    model = OrnsteinUhlenbeck(mu, sigma, theta, X0=X0)

    generator = CppPathGenerator()
    generator = PythonPathGenerator()
    engine = SimulationEngine(model=model, T=1.0, steps=1000, seed=seed, path_generator=generator)
    payoff = EuropeanCall(K=0.0)
    risk = RiskFree()

    pricer = Pricer(engine, payoff, risk)

    time_start = time.time()
    price = pricer.price(samples=int(1e4))
    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.4f} seconds")

    print(f"Estimated price: {price:.4f}")


if __name__ == "__main__":
    main()

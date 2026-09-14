from quantlib_lite.stochastic_models import JumpDiffusion, OrnsteinUhlenbeck
from quantlib_lite.simulation_engine import SimulationEngine
import matplotlib.pyplot as plt
import numpy as np

def main():
    mu = 0.1
    sigma = 0.2
    S0 = 1.5
    lam = 1
    jump_mean = 0.1
    jump_std = 0.2
    model = JumpDiffusion(mu=mu, sigma=sigma, lam=lam, jump_mean=jump_mean, jump_std=jump_std, S0=S0)

    T=1.0
    steps=1000
    
    engine = SimulationEngine(model, T, steps)
    paths = engine.simulate(n_paths=3)

    for path in paths:
        plt.plot(path.times, path.values)

    plt.show()

    plt.clf()
    theta = 0.1
    model = OrnsteinUhlenbeck(mu, sigma, theta, S0=S0)
    engine = SimulationEngine(model, T, steps)
    paths = engine.simulate(n_paths=3)

    for path in paths:
        plt.plot(path.times, path.values)
    plt.show()


if __name__ == "__main__":
    main()


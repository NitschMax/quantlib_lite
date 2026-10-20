from quantlib_lite.stochastic_models import SchwartzSmith
from quantlib_lite.simulation_engine import SimulationEngine
import matplotlib.pyplot as plt
import numpy as np

def main():
    model = SchwartzSmith(
        mu_zeta=0.0,
        sigma_zeta=0.1,
        theta=1.0,
        sigma_xi=0.1,
        zeta_0=1.0,
        xi_0=0.0,
        rho=0.5,
        A=0.1,
        t_w=0.0,
    )

    T = 1.0
    steps = 365
    seed = 7
    n_paths = 9

    engine = SimulationEngine(model, T, steps, seed=seed)
    paths = engine.simulate(n_paths=n_paths)

    for path in paths:
        plt.plot(path.times, path.values)

    plt.xlabel("t")
    plt.ylabel("S_t")
    plt.title("Schwartz-Smith sample paths")
    plt.show()
    plt.clf()

    # t is the grid of valuation dates matching the simulated state's own
    # time axis, T is a single fixed future delivery date broadcast against
    # that grid and against every simulated path in state.
    rng = np.random.default_rng(seed)
    state = model.state_paths_batch(T, steps, n_paths, rng=rng)
    times = model.times(T, steps)

    forward_prices = model.forward(t=times, T=T, state=state)
    for curve in forward_prices:
        plt.plot(times, curve)

    plt.xlabel("t")
    plt.ylabel(f"F(t, T={T})")
    plt.title("Schwartz-Smith forward prices")
    plt.show()
    plt.clf()

    
    times = 0
    deliveries = np.linspace(times, T, 12)
    forward_volatility = model.forward_vol(t=times, T=deliveries)
    plt.plot(deliveries, forward_volatility)

    plt.xlabel("T")
    plt.ylabel(f"Vol(t={times}, T)")
    plt.title("Schwartz-Smith forward volatility")
    plt.show()

if __name__ == "__main__":
    main()

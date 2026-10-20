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

    horizon = 1.0          # simulate the spot price over one year (in years)
    steps = 365            # daily time steps
    seed = 7
    n_paths = 9

    engine = SimulationEngine(model, horizon, steps, seed=seed)
    paths = engine.simulate(n_paths=n_paths)

    for path in paths:
        plt.plot(path.times, path.values)

    plt.xlabel("time t [years]")
    plt.ylabel("spot price S_t")
    plt.title("Schwartz-Smith sample paths")
    plt.show()
    plt.clf()

    # Value one fixed forward contract along each simulated path:
    #   valuation_dates  the dates t at which we price, i.e. the simulation's own
    #                    time grid (one per simulated state on each path)
    #   delivery_date    the single future date T the contract delivers at,
    #                    broadcast against every valuation date and every path
    rng = np.random.default_rng(seed)
    state_paths = model.state_paths_batch(horizon, steps, n_paths, rng=rng)
    valuation_dates = model.times(horizon, steps)
    delivery_date = horizon

    forward_price_paths = model.forward(t=valuation_dates, T=delivery_date, state=state_paths)
    for forward_price_path in forward_price_paths:
        plt.plot(valuation_dates, forward_price_path)

    plt.xlabel("valuation date t [years]")
    plt.ylabel(f"forward price F(t, T={delivery_date})")
    plt.title("Schwartz-Smith forward prices")
    plt.show()
    plt.clf()

    # Term structure of forward volatility seen from today: one valuation date
    # t, many delivery dates T spread from today up to the horizon
    valuation_date = 0.0
    delivery_dates = np.linspace(valuation_date, horizon, 12)
    forward_volatility = model.forward_vol(t=valuation_date, T=delivery_dates)
    plt.plot(delivery_dates, forward_volatility)

    plt.xlabel("delivery date T [years]")
    plt.ylabel(f"forward volatility σ(t={valuation_date}, T)")
    plt.title("Schwartz-Smith forward volatility")
    plt.show()

if __name__ == "__main__":
    main()

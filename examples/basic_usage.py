from quantlib_lite.stochastic_models import GBM
from quantlib_lite.payoff import EuropeanCall
from quantlib_lite.risk_measure import RiskFree
from quantlib_lite.simulation_engine import SimulationEngine
from quantlib_lite import Pricer
from quantlib_lite.path_generator import CppPathGenerator, PythonPathGenerator

def main():
    seed = 1
    model = GBM(mu=0.05, sigma=0.2)
    generator = PythonPathGenerator()
    generator = CppPathGenerator()
    engine = SimulationEngine(model=model, T=1.0, steps=100, seed=seed, path_generator=generator)
    payoff = EuropeanCall(K=0.0)
    risk = RiskFree()

    pricer = Pricer(engine, payoff, risk)

    price = pricer.price(samples=1000)

    print(f"Estimated price: {price:.4f}")


if __name__ == "__main__":
    main()

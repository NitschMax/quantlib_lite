from quantlib_lite.stochastic_models.gbm import GBM
from quantlib_lite.payoff.european_call import EuropeanCall
from quantlib_lite.risk_measure.risk_free import RiskFree
from quantlib_lite.pricer import Pricer

def main():
    model = GBM(mu=0.05, sigma=0.2)
    payoff = EuropeanCall(K=1.0)
    risk = RiskFree()

    pricer = Pricer(model, payoff, risk)

    price = pricer.price(T=1.0, steps=100, samples=1000)

    print(f"Estimated price: {price:.4f}")


if __name__ == "__main__":
    main()

from stochastic_models.gbm import GBM
from payoff.european_call import EuropeanCall
from payoff.asian_call import AsianCall

gbm = GBM(0.05, 0.2)
path = gbm.sample_path(1.0, 10)

print(path)

call = EuropeanCall(K=1.0)
asian = AsianCall(K=1.0)

print(call.evaluate(path))
print(asian.evaluate(path))

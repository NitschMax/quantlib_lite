from quantlib_lite.stochastic_models import JumpDiffusion, OrnsteinUhlenbeck
import matplotlib.pyplot as plt
import numpy as np

def main():
    mu = 0.1
    sigma = 0.2
    lam = 1
    jump_mean = 0.1
    jump_std = 0.2
    model = JumpDiffusion(mu=mu, sigma=sigma, lam=lam, jump_mean=jump_mean, jump_std=jump_std)
    paths = model.sample_paths_batch(T=1.0, steps=1000, n_paths=3)

    for path in paths:
        plt.plot(path.times, path.values)

    plt.show()

    plt.clf()
    theta = 0.1
    model = OrnsteinUhlenbeck(mu, sigma, theta)
    paths = model.sample_paths_batch(T=1.0, steps=1000, n_paths=3)

    for path in paths:
        plt.plot(path.times, path.values)
    plt.show()


if __name__ == "__main__":
    main()


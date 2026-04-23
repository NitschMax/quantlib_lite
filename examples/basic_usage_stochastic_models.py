from quantlib_lite.stochastic_models import JumpDiffusion
import matplotlib.pyplot as plt

def main():
    mu = 0.1
    sigma = 0.2
    lam = 1
    jump_mean = 0.1
    jump_std = 0.2
    model = JumpDiffusion(mu=mu, sigma=sigma, lam=lam, jump_mean=jump_mean, jump_std=jump_std)

    path = model.sample_path(T=1.0, steps=1000)

    plt.plot(path.times, path.values)
    plt.show()



if __name__ == "__main__":
    main()


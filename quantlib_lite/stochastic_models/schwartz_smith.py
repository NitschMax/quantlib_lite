from quantlib_lite.path.path import Path
from .stochastic_model import StochasticModel
from .forward_curve_model import ForwardCurveModel
import numpy as np

class SchwartzSmith(StochasticModel, ForwardCurveModel):
    def __init__(self, mu_zeta, sigma_zeta, theta, sigma_xi, zeta_0, xi_0, rho, A, t_w):
        self.__mu_zeta = float(mu_zeta)     # long-term mean
        self.__sigma_zeta = float(sigma_zeta) # volatitlity brownian motion

        self.__theta = float(theta)         # mean reversion speed
        self.__sigma_xi = float(sigma_xi)   # volatitlity brownian motion

        self.__zeta_0 = float(zeta_0)         # initial value
        self.__xi_0 = float(xi_0)             # initial value
        self.__rho = float(rho)

        self.__A = float(A)                 # seasonal amplitude
        self.__t_w = float(t_w)             # differnce of days/365 to refernce date to adjust the phase to reference date

    @property
    def mu_zeta(self):
        return self.__mu_zeta

    @property
    def sigma_zeta(self):
        return self.__sigma_zeta

    @property
    def theta(self):
        return self.__theta

    @property
    def sigma_xi(self):
        return self.__sigma_xi

    @property
    def zeta_0(self):
        return self.__zeta_0

    @property
    def xi_0(self):
        return self.__xi_0

    @property
    def rho(self):
        return self.__rho

    @property
    def A(self):
        return self.__A

    @property
    def t_w(self):
        return self.__t_w

    @property
    def dimension(self):
        return 2

    def __hash__(self):
        return hash((self.mu_zeta, self.sigma_zeta, self.theta, self.sigma_xi, self.zeta_0, self.xi_0, self.A, self.t_w, self.rho))

    def __eq__(self, other):
        if isinstance(other, SchwartzSmith):
            return (self.mu_zeta, self.sigma_zeta, self.theta, self.sigma_xi, self.zeta_0, self.xi_0, self.A, self.t_w, self.rho) == \
                   (other.mu_zeta, other.sigma_zeta, other.theta, other.sigma_xi, other.zeta_0, other.xi_0, other.A, other.t_w, other.rho)
        else:
            return NotImplemented

    def seasonal_component(self, t):
        return self.A * np.cos(2 * np.pi * (t - self.t_w))

    def state_paths_batch(self, T, steps, n_paths, rng=None):
        if rng == None:
            rng = np.random.default_rng()

        dt = self.dt(T, steps)
        cm = self.covariance_matrix(dt)

        r = cm[0,1]/np.sqrt((cm[0,0]*cm[1,1]) )

        # pre-compute all random numbers for efficiency
        eps_nums = rng.normal(0, 1, (n_paths, steps, 2))
        eps_nums[:, :, 1] = np.sqrt(cm[1, 1])*(r*eps_nums[:, :, 0] + np.sqrt(1-r**2)*eps_nums[:, :, 1])
        eps_nums[:, :, 0] *= np.sqrt(cm[0, 0])


        X = np.zeros((n_paths, steps+1, 2))
        X[:, 0] = [self.zeta_0, self.xi_0]
        exp_theta_dt = np.exp(-self.theta*dt)

        for k in range(steps):
            X[:,k+1,0] = X[:,k,0] + self.mu_zeta * dt + eps_nums[:, k, 0]
            X[:,k+1,1] = X[:,k,1] * exp_theta_dt + eps_nums[:, k, 1]

        return X

    def covariance_matrix(self, dt):
        return np.array([[self.sigma_zeta**2 * dt, self.sigma_zeta*self.sigma_xi * self.rho/self.theta*(-1)*np.expm1(-self.theta*dt)], \
                [self.sigma_zeta*self.sigma_xi * self.rho/self.theta*(-1)*np.expm1(-self.theta*dt), self.sigma_xi**2/self.theta/2*(-1)*np.expm1(-2*self.theta*dt)]])

    def sample_paths_batch(self, T, steps, n_paths, rng=None):
        X = self.state_paths_batch(T, steps, n_paths, rng=rng)
        times = self.times(T, steps)

        S_t = X[:,:,0] + X[:,:,1] 
        S_t += self.seasonal_component(times)
        S_t = np.exp(S_t)
        return [Path(times, x) for x in S_t]

    def log_forward(self, t, T, state):
        state, t, T = np.asarray(state, float), np.asarray(t, float), np.asarray(T, float)
        tau = T - t
        grid = (1,) * T.ndim
        return state[..., 1].reshape(state.shape[:-1] + grid) * np.exp(-self.theta*tau) + state[..., 0].reshape(state.shape[:-1] + grid) + self.mu_zeta * tau + self.seasonal_component(T) + self.covariance_matrix(tau).sum(axis=(0, 1))/2

    def forward(self, t, T, state):
        return np.exp(self.log_forward(t, T, state))

    def forward_vol(self, t, T):
        tau = T - t
        return np.sqrt(self.sigma_zeta**2 + self.sigma_xi**2 * np.exp(-2*self.theta*tau) + 2 * self.rho * self.sigma_zeta * self.sigma_xi * np.exp(-self.theta * tau))

    def initial_state(self):
        return np.array([self.zeta_0, self.xi_0])
















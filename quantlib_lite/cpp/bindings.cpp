#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <cmath>

std::string hello() {
	return "Hello from C++";
}

std::vector<double> gbm_path(double mu, double sigma, double T, int steps, unsigned int seed) {
	double dt = T / steps;
	std::vector<double> path(steps + 1);
	path[0] = 1.0; // Initial value
	
	std::mt19937 rng(seed);
	std::normal_distribution<double> normal(0.0, 1.0);

	double W = 0.0;
	double dt_sqrt = std::sqrt(dt);
	for (int i = 1; i <= steps; ++i){
		W += dt_sqrt * normal(rng);
		double t = i * dt;
		path[i] = std::exp((mu - 0.5 * sigma * sigma) * t + sigma * W);
	}
	return path;
}

std::vector<double> jump_diffusion_path(double mu, double sigma, double lam, double jump_mean, double jump_std, double T, int steps, unsigned int seed) {
	double dt = T / steps;

	std::mt19937 rng(seed);
	std::normal_distribution<double> normal(0.0, 1.0);

	double dt_sqrt = std::sqrt(dt);
	std::vector<double> W(steps+1);
	W[0] = 0;
	for (int i = 1; i <= steps; ++i){
		W[i] = W[i-1] + dt_sqrt * normal(rng);
	}

	std::poisson_distribution<int> poisson(lam * dt);  // lambda = rate parameter

	std::vector<double> dN(steps+1);
	dN[0] = 0;
	for (int i = 1; i <= steps; ++i){
		dN[i] = poisson(rng);
	}

	std::vector<double> jumps_acc(steps+1);
	for (int i = 1; i <= steps; ++i){
		jumps_acc[i] = jumps_acc[i-1];
		if (dN[i] > 0) {
			jumps_acc[i] += jump_mean * dN[i] + normal(rng) * jump_std * std::sqrt(dN[i]);
		}
	}
	
	double k = std::exp(jump_mean + 0.5 * jump_std * jump_std) - 1;
	std::vector<double> path(steps+1);
	path[0] = 1.0;
	for (int i = 1; i <= steps; ++i){
		double t = i * dt;
        	path[i] = std::exp((mu - 0.5 * sigma * sigma - k * lam) * t + sigma * W[i] + jumps_acc[i]);
	}

	return path;
}


PYBIND11_MODULE(quantlib_lite_cpp, m) {
	m.doc() = "C++ accelerated bindings for quantlib_lite";
	m.def("hello", &hello, "A function returning a greeting from C++");
	m.def("gbm_path", &gbm_path, "A C++ implementation of GBM path generation");
	m.def("jump_diffusion_path", &jump_diffusion_path, "A C++ implementation of jump diffusion path generation");
}

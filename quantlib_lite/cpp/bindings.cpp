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

PYBIND11_MODULE(quantlib_lite_cpp, m) {
	m.doc() = "C++ accelerated bindings for quantlib_lite";
	m.def("hello", &hello, "A function returning a greeting from C++");
	m.def("gbm_path", &gbm_path, "A C++ implementation of GBM path generation");
}

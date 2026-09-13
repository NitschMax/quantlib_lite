import quantlib_lite_cpp as qlc
from .path_generator import PathGenerator
from quantlib_lite.stochastic_models import GBM
from quantlib_lite.path import Path

class CppPathGenerator(PathGenerator):
    def generate(self, model, T, steps, n_paths, rng):
        if not isinstance(model, GBM):
            raise NotImplementedError(f"No C++ implementation for this model yet available!")

        times = model.times(T, steps)
        paths = []

        for idx in range(n_paths):
            seed = int(rng.integers(0, 2*63))
            values = qlc.gbm_path(model.mu, model.sigma, T, steps, seed)

            paths.append(Path(times, values))

        return paths
            

        


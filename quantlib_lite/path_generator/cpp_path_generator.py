import quantlib_lite_cpp as qlc
from .path_generator import PathGenerator
from quantlib_lite.stochastic_models import GBM, JumpDiffusion
from quantlib_lite.path import Path

class CppPathGenerator(PathGenerator):
    def generate(self, model, T, steps, n_paths, rng):
        if not (isinstance(model, GBM) or isinstance(model, JumpDiffusion) ):
            raise NotImplementedError(f"No C++ implementation for this model yet available!")

        times = model.times(T, steps)
        paths = []

        for idx in range(n_paths):
            seed = int(rng.integers(0, 2**32))
            if isinstance(model, GBM):
                values = qlc.gbm_path(model.mu, model.sigma, T, steps, seed)
            elif isinstance(model, JumpDiffusion):
                values = qlc.jump_diffusion_path(model.mu, model.sigma, model.lam, model.jump_mean, model.jump_std, T, steps, seed)

            paths.append(Path(times, values))

        return paths
            

from .path_generator import PathGenerator

class PythonPathGenerator(PathGenerator):
    def generate(self, model, T, steps, n_paths, rng):
        return model.sample_paths_batch(T, steps, n_paths, rng)

from abc import ABC, abstractmethod

class PathGenerator(ABC):

    @abstractmethod
    def generate(self, model, T, steps, n_paths, rng):
        """This class is a wrapper to delegate the simulation logic either to the native python implementation or a C++ implementation.

        Parameters
        ----------
        model : object
            The model to be used for path generation.
        T : float
            The time horizon for the path generation.
        steps : int
            The number of time steps for the path generation.
        n_paths : int
            The number of paths to be generated.
        rng : object
            The random number generator to be used for path generation.

        Returns
        -------
        paths : lst
            The generated paths as a list
        """


from __future__ import annotations

import numpy as np
from ray.rllib.utils.spaces.simplex import Simplex


class CustomSimplex(Simplex):
    """Represents a d - 1 dimensional Simplex in R^d.

    That is, all coordinates are in [0, 1] and sum to 1.

    Additionally one can specify the underlying distribution of
    the simplex as a Dirichlet distribution by providing concentration
    parameters. By default, sampling is uniform, i.e. concentration is
    all 1s.

    Example usage:
    self.action_space = spaces.Simplex(3, concentration=[1.0, 1.0, 1.0])
        --> Dirichlet in d dimensions with a uniform concentration
    """

    def __init__(self, dim: int, concentration=None):
        self.shape = (dim,)
        self.dtype = np.float32
        self._np_random = None

        self.dim = dim
        self.concentration = concentration if concentration else [1] * self.dim

    def sample(self):
        return np.random.dirichlet(self.concentration).astype(self.dtype)

    def contains(self, x):
        return x.dim == self.dim

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Simplex({}; {})".format(self.dim, self.concentration)

    def __eq__(self, other):
        return np.allclose(self.concentration, other.concentration) and self.dim == other.dim

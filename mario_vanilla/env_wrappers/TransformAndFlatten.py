from gym import ObservationWrapper
import numpy as np
from numpy import ndarray

class TransformAndFlatten(ObservationWrapper):
    def __init__(self, env, positioner, dim):
        super().__init__(env)
        self.positioner = positioner
        self.dim = dim

    def observation(self, observation) -> ndarray:
        """Transforms the observation to a ndarray of shape self.dim.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """

        cell_list = self.positioner.position(observation)

        return self.convert_ASP_cells_to_matrix(cell_list, self.dim)

    def convert_ASP_cells_to_matrix(self, cell_list: list, dim) -> ndarray:

        matrix = np.zeros(dim, dtype=int)

        for cell in cell_list:
            row, col, val = map(int, cell.strip('cell().').split(','))
            matrix[row, col] = val

        return matrix
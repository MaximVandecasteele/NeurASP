
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








        # # expecting input to be a DataFrame with first column 'name'.
        # # this column can be dropped. It will only be relevant in Phase2
        # positions = observation.drop(['name'], axis=1).to_numpy().copy()
        #
        # # make a 1D vector that fits the mlpPolicy
        # flattened = positions.reshape(-1)
        # # padding the array with negative 1
        # padded = np.pad(flattened, (0, self.dim - (positions.shape[0] * positions.shape[1])), 'constant',
        #                 constant_values=(-1))
        #
        # return padded

    def convert_ASP_cells_to_matrix(self, cell_list: list, dim) -> ndarray:

        matrix = np.zeros(dim, dtype=int)

        for cell in cell_list:
            row, col, val = map(int, cell.strip('cell().').split(','))
            matrix[row, col] = val

        return matrix
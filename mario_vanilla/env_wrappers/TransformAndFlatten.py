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

        result = self.one_hot_encode(matrix,7)

        return result

    # def one_hot_encode(self, matrix, num_classes):
    #     # Flatten the matrix to a vector
    #     flat_matrix = matrix.flatten()
    #
    #     # Create a one-hot encoded matrix of zeros
    #     one_hot_matrix = np.zeros((flat_matrix.size, num_classes))
    #
    #     # Set the appropriate element to one
    #     one_hot_matrix[np.arange(flat_matrix.size), flat_matrix] = 1
    #
    #     # Reshape back to the original matrix shape with an added dimension for the one-hot encoding
    #     one_hot_matrix = one_hot_matrix.reshape(*matrix.shape, num_classes)
    #
    #     return one_hot_matrix

    def one_hot_encode(self, matrix, num_classes):
        # Get the height and width of the original matrix
        height, width = matrix.shape

        # Initialize the one-hot encoded matrix with zeros
        one_hot_matrix = np.zeros((num_classes, height, width), dtype=np.int32)

        # Iterate over the matrix and set the appropriate index to 1
        for i in range(height):
            for j in range(width):
                class_index = matrix[i, j]
                one_hot_matrix[class_index, i, j] = 1

        return one_hot_matrix

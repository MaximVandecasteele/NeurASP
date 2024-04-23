from gym import ObservationWrapper
import numpy as np
from numpy import ndarray


class OneHot(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)


    def observation(self, observation) -> ndarray:

        return self.one_hot_encode(observation, num_classes= 7)

    def one_hot_encode(self, tensor, num_classes):
            # tensor shape is assumed to be (4, 15, 16)
            # Initialize the one-hot encoded tensor with zeros
            one_hot_encoded = np.zeros((num_classes, *tensor.shape))

            # Iterate over each element and set the appropriate class index
            for depth in range(tensor.shape[0]):
                for height in range(tensor.shape[1]):
                    for width in range(tensor.shape[2]):
                        # Get the class index from the original tensor
                        class_index = tensor[depth, height, width]
                        # Set the class index to 1 for the one-hot encoded tensor
                        one_hot_encoded[class_index, depth, height, width] = 1

            return one_hot_encoded

        # Example usage:
        # Assuming 'input_tensor' is your 4 x 15 x 16 input tensor with class indices
        # And 'num_classes' is the total number of classes
        # one_hot_encoded_tensor = one_hot_encode(input_tensor, num_classes)

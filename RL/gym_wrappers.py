
import gym
import collections
import numpy as np
import cv2

from RL.symbolic_components.positioner import Positioner
from RL.symbolic_components.detector import Detector
import torch




##### Setting up Mario environment #########
#  skipframe
#  redefines step method and reset method
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


if torch.backends.mps.is_available():
    config = {
            "detector_model_path": '/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
            "detector_label_path": '/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/models/data.yaml',
            "positions_asp": '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/positions.lp',
            "show_asp": '/Users/maximvandecasteele/PycharmProjects/NeurASP/RL/asp/show.lp'
        }
elif torch.cuda.is_available():
    config = {
            "detector_model_path": '/home/stefaan/local/python/NeurASP/Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
            "detector_label_path": '/home/stefaan/local/python/NeurASP/Object_detector/models/data.yaml',
            "positions_asp": '/home/stefaan/local/python/NeurASP/RL/asp/positions.lp',
            "show_asp": '/home/stefaan/local/python/NeurASP/RL/asp/show.lp'
        }

class ProcessFrame(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    And applies semantic segmentation if set to. Otherwise, uses grayscale normal frames.
    Returns numpy array
    """
    def __init__(self, input_type, env=None):
        super(ProcessFrame, self).__init__(env)
        self.input_type = input_type

        if self.input_type == 'asp':
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(15, 16, 1), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        if self.input_type == 'asp':
            self.detector = Detector(config)
            self.positioner = Positioner(config)

    def observation(self, obs):
        return self.process(obs, self.input_type)

    def process(self, frame, input_type):
        if frame.size == 240 * 256 * 3:
            img_og = np.reshape(frame, [240, 256, 3]).astype(np.uint8)
            # If using semantic segmentation:
            if input_type == 'ss':
                img = self.segmentator.segment_labels(img_og)

                # Normalize labels so they are evenly distributed in values between 0 and 255 (instead of being  0,1,2,...)
                img = np.uint8(img * 255 / 6)

                # Re-scale image to fit model.
                resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_NEAREST)
                x_t = resized_screen[18:102, :]
                x_t = np.reshape(x_t, [84, 84, 1])

            elif input_type == 'asp':

                positions = self.detector.detect(frame)
                cell_list = self.positioner.position(positions)
                img = self.convert_ASP_cells_to_matrix(cell_list, (15, 16, 1))
                x_t = np.uint8(img * 255 / 6)

            else:
                img = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                # Re-scale image to fit model.
                resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_NEAREST)
                x_t = resized_screen[18:102, :]
                x_t = np.reshape(x_t, [84, 84, 1])

        else:
            assert False, "Unknown resolution."

        return x_t.astype(np.uint8)

    def convert_ASP_cells_to_matrix(self, cell_list: list, dim):

        matrix = np.zeros(dim, dtype=int)

        for cell in cell_list:
            row, col, val = map(int, cell.strip('cell().').split(','))
            matrix[row, col, 0] = val

        return matrix


#Defines a float 32 image with a given shape and shifts color channels to be the first dimension (for pytorch)
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


#Stacks the latests observations along channel dimension
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    #buffer frames.
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer
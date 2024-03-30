import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace


from gym.vector.utils import spaces
import numpy as np

from stable_baselines3_master.stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env_wrappers.SkipFrame import SkipFrame
from env_wrappers.DetectObjects import DetectObjects
from env_wrappers.TransformAndFlatten import TransformAndFlatten


# class SkipFrame(Wrapper):
#     def __init__(self, env, skip):
#         super().__init__(env)
#         self.skip = skip
#
#     def step(self, action):
#         total_reward = 0.0
#         done = False
#         for _ in range(self.skip):
#             next_state, reward, done, trunc, info = self.env.step(action)
#             total_reward += reward
#             if done:
#                 break
#         return next_state, total_reward, done, trunc, info
#

def apply_wrappers(env, config):
    # nes_py bugfix
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=config["skip"]) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=config['stack_size'], lz4_compress=True) # May need to change lz4_compress to False if issues arise
    return env



def apply_ASP_wrappers(env, config, detector, positioner):
    # nes_py bugfix
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    # 1. Simplify the controls
    env = JoypadSpace(env, RIGHT_ONLY)
    # 2. There is not much difference between frames, so take every fourth
    env = SkipFrame(env, skip=config["skip"])  # Num of frames to apply one action to

    # 3. Detect, position and reduce dimension
    env = DetectObjects(env, detector=detector)  # intercept image and convert to object positions
    # env = PositionObjects(env, positioner=positioner)  # intercept image and convert to object positions
    env = TransformAndFlatten(env, positioner, dim=config["observation_dim"])
    # 4. Wrap inside the Dummy Environment
    # env = DummyVecEnv([lambda: env])

    # 5. Stack the frames
    env = FrameStack(env, num_stack=config['stack_size'], lz4_compress=True)
    # env = VecFrameStack(env, config["stack_size"], channels_order='last')


    return env
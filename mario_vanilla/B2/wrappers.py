from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from mario_vanilla import (SkipFrame)
from mario_vanilla import DetectObjects
from mario_vanilla import TransformAndFlatten

def apply_ASP_wrappers(env, config, detector, positioner):
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
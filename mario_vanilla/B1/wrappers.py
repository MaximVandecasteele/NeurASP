from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from mario_vanilla import SkipFrame



def apply_wrappers(env, config):
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=config["skip"]) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=config['stack_size'], lz4_compress=True) # May need to change lz4_compress to False if issues arise
    return env

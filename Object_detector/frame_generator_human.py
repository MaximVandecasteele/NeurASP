import pygame

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.utils.play import play

from nes_py.wrappers import JoypadSpace
from Environment.gym_wrappers import apply_img_capture_wrappers, make_neurasp_env

import warnings
warnings.filterwarnings('ignore')

mapping = {
    # ...
    (): 0,
    (pygame.K_f,): 1,
    (pygame.K_f, pygame.K_i): 2,
    (pygame.K_f, pygame.K_j): 3,
    (pygame.K_f, pygame.K_i, pygame.K_j): 4,
    (pygame.K_i, ): 5,
    (pygame.K_d,): 6,
    (pygame.K_d, pygame.K_i): 7,
    (pygame.K_d, pygame.K_j): 8,
    (pygame.K_d, pygame.K_j, pygame.K_i): 9,
    (pygame.K_x,): 10,
    (pygame.K_e,): 11
    # ...
}

ENV_NAME = 'SuperMarioBros-1-1-v0'

# if True, you get a game window in order to play the game.
DISPLAY = True

# if False, you only play the game
CAPTURE_FRAMES = False
# if True, you store build a dataset for NeurASP training purposes
BUILD_DATASET = False

env = gym_super_mario_bros.make('SuperMarioBros-'+'1-1'+'-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

if CAPTURE_FRAMES:
    env = apply_img_capture_wrappers(env=env, directory='./frames/', env_name=ENV_NAME)
elif BUILD_DATASET:
    env = make_neurasp_env(env, 'symbols_naive.csv', 'tensors_naive.pkl')

play(env, keys_to_action=mapping)



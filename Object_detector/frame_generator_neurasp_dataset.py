import pygame

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.utils.play import play

from nes_py.wrappers import JoypadSpace
from Object_detector.wrappers import make_neurasp_env
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
    (pygame.K_e,): 11,
    # ...
}

# Create gym environment

env = gym_super_mario_bros.make('SuperMarioBros-'+'1-1'+'-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = make_neurasp_env(env)


# Play the game

play(env, keys_to_action=mapping)



#Code modified from [on the Paperspace blog](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/).

## Install the following if on a new instance, otherwise they'll ship with the container.
# !pip install nes-py==0.2.6
# !pip install gym-super-mario-bros
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y

import csv
from ast import parse
from turtle import back
import torch
import torch.nn as nn
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from torch.serialization import save
from tqdm import tqdm
import pickle 

from gym_super_mario_bros.actions import RIGHT_ONLY

import gym
import numpy as np
import collections 
import cv2
import matplotlib.pyplot as plt
from IPython import display
from segmentator import Segmentator
from gym_wrappers import MaxAndSkipEnv, ProcessFrame, ImageToPyTorch, ScaledFloatFrame, BufferWrapper
from DQN_network_vanilla import DQNSolver
from DQNAgent import DQNAgent


# Argparse

import argparse
import os

config = {
    'vis': True,
    'level': '1-1',
    # asp or rgb
    'input_type': 'asp',
    'inference_type': 'pure',
    'train': True,
    'exp_r': 0.1,
    'num_runs': 5,
    'epochs': 100,
    'working_dir': 'Models_asp/run_1/',
    'model': 'run3_',
    'pretrained_weights': True,
    'load_experience_replay': False,
    'save_experience_replay': False,
}
asp = False
if config['input_type'] == 'asp':
    asp = True

### Run settings.
training = config['train']
vis = config['vis']
level = config['level']
inference_type = config['inference_type']

input_type = config['input_type']

exp_rate = config['exp_r']

epochs = config['epochs']

#Model loading
savepath = config['working_dir']

pretrained_weights = config['pretrained_weights']




# ##### Setting up Mario environment #########
#Create environment (wrap it in all wrappers)
def make_env(env):
    env = MaxAndSkipEnv(env)
    #print(env.observation_space.shape)
    env = ProcessFrame(input_type, env)
    #print(env.observation_space.shape)

    env = ImageToPyTorch(env)
    #print(env.observation_space.shape)

    env = BufferWrapper(env, 6)
    #print(env.observation_space.shape)

    env = ScaledFloatFrame(env)
    #print(env.observation_space.shape)

    return JoypadSpace(env, RIGHT_ONLY) #Fixes action sets

def make_asp_env(env):
    env = MaxAndSkipEnv(env)
    #print(env.observation_space.shape)
    env = ProcessFrame(input_type, env)
    #print(env.observation_space.shape)

    env = ImageToPyTorch(env)
    #print(env.observation_space.shape)

    env = BufferWrapper(env, 6)
    #print(env.observation_space.shape)

    env = ScaledFloatFrame(env)
    #print(env.observation_space.shape)

    return JoypadSpace(env, RIGHT_ONLY) #Fixes action sets

def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]

#Shows current state (as seen in the emulator, not segmented)
def show_state(env, ep=0, info=""):
    cv2.imshow("Output!",env.render(mode='rgb_array')[:,:,::-1]) #Display using opencv
    cv2.waitKey(1)





def run(asp, pretrained):
   
    env = gym_super_mario_bros.make('SuperMarioBros-'+level+'-v0') #Load level

    if asp:
        env = make_asp_env(env) # Wraps the environment so that frames are 15x16 ASP frames
    else:
        env = make_env(env)  # Wraps the environment so that frames are grayscale / segmented

    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    pretrained_model_name = config['model']
    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=4000,
                     batch_size=16,
                     gamma=0.90,
                     lr=0.00025,
                     dropout=0.,
                     exploration_max=exp_rate,
                     exploration_min=exp_rate,
                     exploration_decay=0.99,
                     pretrained=pretrained,
                     savepath=savepath,
                     load_exp_rep=False,
                     pretrained_model_name=pretrained_model_name,
                     asp=asp)

    #Reset environment
    env.reset()

    #Each iteration is an episode (epoch)
    for ep_num in range(epochs):
        #Reset state and convert to tensor
        state = env.reset()
        state = torch.Tensor(np.array([state]))

        #Set episode total reward and steps
        total_reward = 0
        steps = 0
        #Until we reach terminal state
        while True:
            #Visualize or not
            if vis:
                show_state(env, ep_num)
            
            #What action would the agent perform
            action = agent.act(state)
            #Increase step number
            steps += 1
            #Perform the action and advance to the next state
            state_next, reward, terminal, info = env.step(int(action[0]))
            #Update total reward
            total_reward += reward
            #Change to next state
            state_next = torch.Tensor(np.array([state_next]))
            #Change reward type to tensor (to store in ER)
            reward = torch.tensor(np.array([reward])).unsqueeze(0)
            #Is the new state a terminal state?
            terminal = torch.tensor(np.array([int(terminal)])).unsqueeze(0)
            #Update state to current one
            state = state_next
            #Write to tensorboard Reward and position
            if terminal == True:
                break #End episode loop


    
    env.close()



if __name__ == '__main__':
    run(asp=asp, pretrained=pretrained_weights)


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
    'vis': False,
    'level': '1-1',
    'tensorboard' : True,
    # asp or rgb
    'input_type': 'asp',
    'inference_type': 'pure',
    'train': False,
    'max_exp_r': 1.0,
    'min_exp_r': 0.02,
    'num_runs': 5,
    'epochs': 100,
    'working_dir': 'training_run_baseline/Models_asp/',
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
use_tensorboard = config['tensorboard']
inference_type = config['inference_type']
min_exp_r = config['min_exp_r']
max_exp_r = config['max_exp_r']

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter


input_type = config['input_type']

##Training settings:
if training ==  False:
    if inference_type == 'pure':
        max_exploration_rate = 0.02
        min_exploration_rate = 0.02
    else:
        max_exploration_rate = min_exp_r
        min_exploration_rate = min_exp_r
else:
    max_exploration_rate = max_exp_r
    min_exploration_rate = min_exp_r

epochs = config['epochs']

#Model loading
savepath = config['working_dir']

pretrained_weights = config['pretrained_weights']


#What to do with experience replay
load_exp_rep = config['load_experience_replay']
save_exp_rep = config['save_experience_replay']


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

    csv_file_path = f'data_performance_{input_type}.csv'

    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            pass  # Create an empty file
    # Appending to CSV
    with open(csv_file_path, mode='a+', newline='') as file:
        writer = csv.writer(file)

        for m in range(config['num_runs']):
            for e in range(100, 5001, 100):
                pretrained_model_name = f'run_{m + 1}/run{m + 1}_{e}best_performer_'
                agent = DQNAgent(state_space=observation_space,
                                 action_space=action_space,
                                 max_memory_size=4000,
                                 batch_size=16,
                                 gamma=0.90,
                                 lr=0.00025,
                                 dropout=0.,
                                 exploration_max=max_exploration_rate,
                                 exploration_min=min_exploration_rate,
                                 exploration_decay=0.99,
                                 pretrained=pretrained,
                                 savepath=savepath,
                                 load_exp_rep=load_exp_rep,
                                 pretrained_model_name=pretrained_model_name,
                                 asp=asp)

                #Reset environment
                env.reset()

                #If using tensorboard initialize summary_writer
                if use_tensorboard == True:
                    tensorboard_writer = SummaryWriter(f'training_run_baseline/tensorboard_evaluation/{input_type}_performance/run{m + 1}_model{e}')

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
                        #Get x_position (used only with tensorboard)
                        if use_tensorboard == True:
                            x_pos = info['x_pos']
                            flag = info['flag_get']
                        #Is the new state a terminal state?
                        terminal = torch.tensor(np.array([int(terminal)])).unsqueeze(0)
                        #Update state to current one
                        state = state_next
                        #Write to tensorboard Reward and position
                        if use_tensorboard and terminal:
                            tensorboard_writer.add_scalar('Reward',
                                            total_reward ,
                                            ep_num)
                            tensorboard_writer.add_scalar('Position',
                                    x_pos ,
                                    ep_num)
                            break
                        elif terminal == True:
                            break #End episode loop
                    data = [[m + 1, e, ep_num + 1, total_reward, flag]]
                    for row in data:
                        writer.writerow(row)
                    print(f"{m+1}-{e}-{ep_num + 1}: {total_reward}")
    env.close()



if __name__ == '__main__':

    run(asp=asp, pretrained=pretrained_weights)


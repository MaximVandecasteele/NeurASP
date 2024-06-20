#Code modified from [on the Paperspace blog](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/).

## Install the following if on a new instance, otherwise they'll ship with the container.
# !pip install nes-py==0.2.6
# !pip install gym-super-mario-bros
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y

import csv
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from gym_super_mario_bros.actions import RIGHT_ONLY

import numpy as np
import cv2
from Environment.gym_wrappers import MaxAndSkipEnv, ProcessFrame, ImageToPyTorch, ScaledFloatFrame, BufferWrapper
from RL.DQNAgent import DQNAgent
from Configuration.config import config_evaluation
from Environment.gym_wrappers import make_env

# Argparse

import os

config = config_evaluation

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


if inference_type == 'pure':
    max_exploration_rate = 0.02
    min_exploration_rate = 0.02
else:
    max_exploration_rate = min_exp_r
    min_exploration_rate = min_exp_r

epochs = config['epochs']

#Model loading
savepath = config['working_dir']

pretrained_weights = config['pretrained_weights']


#What to do with experience replay
load_exp_rep = config['load_experience_replay']
save_exp_rep = config['save_experience_replay']

def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]

#Shows current state (as seen in the emulator, not segmented)
def show_state(env, ep=0, info=""):
    cv2.imshow("Output!",env.render(mode='rgb_array')[:,:,::-1]) #Display using opencv
    cv2.waitKey(1)

def run(asp, pretrained):
   
    env = gym_super_mario_bros.make('SuperMarioBros-'+level+'-v0') #Load level

    env = make_env(env, input_type)  # Wraps the environment so that frames are grayscale / segmented

    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    csv_file_path = f'data_generalization_{input_type}_2.csv'

    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            pass  # Create an empty file
    # Appending to CSV
    with open(csv_file_path, mode='a+', newline='') as file:
        writer = csv.writer(file)

        for m in range(config['num_runs']):

                pretrained_model_name = f'neurasp_0_model.pt'
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
                    tensorboard_writer = SummaryWriter(f'training_run_neurasp_post/tensorboard_generalization/{input_type}_generalization/run{m + 1}_model0')

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
                    data = [[m + 1, ep_num + 1, total_reward, flag]]
                    for row in data:
                        writer.writerow(row)
                    print(f"{m+1}-{ep_num + 1}: {total_reward}")
    env.close()



if __name__ == '__main__':

    run(asp=asp, pretrained=pretrained_weights)


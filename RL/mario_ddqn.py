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
parser = argparse.ArgumentParser()
#Run settings:
parser.add_argument("-vis","--visualization",help="Visualize the game screen",action='store_true', default=False)
parser.add_argument("--level",help="What level to play",type=str,default="1-1")
parser.add_argument("--tensorboard",help="Log to tensorboard. Default = True",default=True)
parser.add_argument("--run_name",help="A name for the run. Used in tensorboard. Defaults to Test",type=str,default="Test")
parser.add_argument("-it","--input_type",help="Wether to use semantic segmentation (ss), (asp) or normal RGB frames (rgb).",type=str,default="asp")
parser.add_argument("-inf","--inference_type",help="Wether to run inference with no randomness or maintain a small randomness amount. Can be pure or random",type=str,default='pure')

#Training settings
parser.add_argument("-t","--train",help="Training mode",action='store_true',default=True)
parser.add_argument("--max_exp_r",help="Max exploration rate. Defaults to 1", type=float,default=1.0)
parser.add_argument("--min_exp_r",help="Min_exp_rate minimum value for exploration rate",type=float,default=0.02) #if set to 0, it will stop exploring and probably plateau.
parser.add_argument("-e","--epochs",help="Amount of epochs to train for.",type=int,default=1000)
parser.add_argument("-bue","--backup_epochs",help="Backups every e epochs.",type=int,default=100)
parser.add_argument("-sgm","--save_good_model",help="If a model outperforms X times in a row, save it just in case.",type=int,default=-1)

#Model saving and loading
parser.add_argument("-wd","--working_dir",help='Where will files be stored to and loaded from',type=str,default="Models_asp_mac") #required=True
parser.add_argument("-pt","--pretrained_weights",help="Use a pretrained model. Defaults to False",action='store_true', default=False)
parser.add_argument("-mn","--model_name",help="Name of the model to load (if different from default)",type=str, default="")


#Other files that can be saved:
parser.add_argument("--load_experience_replay",help="Load a previously saved experience replay dataset. Defaults to false",type=bool,default=False)
parser.add_argument("--save_experience_replay",help="Save the experience replay dataset.Defaults to False. WARNING: Test to 1 or 2 epochs before fully training, or it may give error when saving.",type=bool,default=False)

#parser.add_argument("-h","--help",help="Prints this command and exit",action="store_true")

args = parser.parse_args()

### Run settings.
training = args.train
vis = args.visualization
level = args.level
use_tensorboard = args.tensorboard

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter

run_name = args.run_name

if args.input_type == 'ss':
    input_type = 'ss'
    # segmentator = Segmentator() #Load segmentation model
elif args.input_type == 'asp':
    input_type = 'asp'
else:
    input_type = 'rgb'

##Training settings:
if args.train ==  False:
    if args.inference_type == 'pure':
        max_exploration_rate = 0
        min_exploration_rate = 0
    else:
        max_exploration_rate = args.min_exp_r
        min_exploration_rate = args.min_exp_r
else:
    max_exploration_rate = args.max_exp_r
    min_exploration_rate = args.min_exp_r

epochs = args.epochs

if args.backup_epochs > 0:
    backup_interval = args.backup_epochs
else:
    backup_interval = -1

save_good_model = args.save_good_model
#Model saving and loading
#Is there a directory for models? otherwise create it
dir_exist = os.path.exists(args.working_dir) and os.path.isdir(args.working_dir)
if not dir_exist:
    os.mkdir(args.working_dir)
savepath = args.working_dir+'/'

pretrained_weights = args.pretrained_weights
pretrained_model_name = args.model_name

#What to do with experience replay
load_exp_rep = args.load_experience_replay
save_exp_rep = args.save_experience_replay


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

def run(asp, training_mode, pretrained):
   
    env = gym_super_mario_bros.make('SuperMarioBros-'+level+'-v0') #Load level

    if asp:
        env = make_asp_env(env) # Wraps the environment so that frames are 15x16 ASP frames
    else:
        env = make_env(env)  # Wraps the environment so that frames are grayscale / segmented

    observation_space = env.observation_space.shape
    action_space = env.action_space.n

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

    #Store rewards and positions
    total_rewards = []
    ending_positions = []
    
    #If using tensorboard initialize summary_writer
    if use_tensorboard == True:
        tensorboard_writer = SummaryWriter('tensorboard/'+run_name+"_labels")

    max_reward = 0
    current_counter = save_good_model

    #Each iteration is an episode (epoch)
    for ep_num in tqdm(range(epochs)):
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

            #Is the new state a terminal state?
            terminal = torch.tensor(np.array([int(terminal)])).unsqueeze(0)

            ### Actions performed while training:
            if training_mode:
                #If the episode is finished:
                if terminal:
                    ######################### Model backup section#############################
                    save = False
                    #Backup interval.
                    if ep_num % backup_interval == 0 and ep_num > 0:
                        save = True
                    #Update max reward
                    if max_reward < total_reward:
                        max_reward = total_reward
                    else:
                        #If the model beats a minimum performance level and beats at least a 70% of the max reward it may be a "good" model
                        if (total_reward > 0.7*max_reward and total_reward > 1000):
                            current_counter = current_counter - 1 #reduce counter by one

                            if current_counter == 0: #if the counter reaches 0, model has outperformed X times in a row, save it.
                                save = True
                            elif current_counter < 0: #if the counter is negative, then this saving method is disabled
                                current_counter = -1
                        else:
                            current_counter = save_good_model #if it doesnt, restart coutner.
                    

                    # Save model backup
                    if save == True:
                        with open(savepath+"bp_ending_position.pkl", "wb") as f:
                            pickle.dump(agent.ending_position, f)
                        with open(savepath+"bp_num_in_queue.pkl", "wb") as f:
                            pickle.dump(agent.num_in_queue, f)
                        with open(savepath+run_name+"_bp_total_rewards.pkl", "wb") as f:
                            pickle.dump(total_rewards, f)
                        with open(savepath+run_name+"_bp_ending_positions.pkl", "wb") as f:
                            pickle.dump(ending_positions, f)   

                        torch.save(agent.local_net.state_dict(),savepath+ str(ep_num)+"best_performer_dq1.pt")
                        torch.save(agent.target_net.state_dict(),savepath+ str(ep_num)+"best_performer_dq2.pt")
                            
                        if save_exp_rep: #If save experience replay is on.
                            print("Saving Experience Replay....")
                            torch.save(agent.STATE_MEM,  savepath+"bp_STATE_MEM.pt")
                            torch.save(agent.ACTION_MEM, savepath+"bp_ACTION_MEM.pt")
                            torch.save(agent.REWARD_MEM, savepath+"bp_REWARD_MEM.pt")
                            torch.save(agent.STATE2_MEM,savepath+ "bp_STATE2_MEM.pt")
                            torch.save(agent.DONE_MEM,   savepath+"bp_DONE_MEM.pt")

                ######################### End of Model Backup Section #################################
                #Add state to experience replay "dataset"
                agent.remember(state, action, reward, state_next, terminal)
                #Learn from experience replay.
                agent.experience_replay()

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

        #Store rewards and positions. Print total reward after episode.
        total_rewards.append(total_reward)
        ending_positions.append(agent.ending_position)
        print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))
            
    
    if training_mode:
        with open(savepath+"ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open(savepath+"num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open(savepath+run_name+"_total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        with open(savepath+run_name+"_ending_positions.pkl", "wb") as f:
            pickle.dump(ending_positions, f)
        torch.save(agent.local_net.state_dict(),savepath+ "dq1.pt")
        torch.save(agent.target_net.state_dict(),savepath+ "dq2.pt")


        if save_exp_rep:
            print("Saving Experience Replay....")
            torch.save(agent.STATE_MEM,  savepath+"STATE_MEM.pt")
            torch.save(agent.ACTION_MEM, savepath+"ACTION_MEM.pt")
            torch.save(agent.REWARD_MEM, savepath+"REWARD_MEM.pt")
            torch.save(agent.STATE2_MEM,savepath+ "STATE2_MEM.pt")
            torch.save(agent.DONE_MEM,   savepath+"DONE_MEM.pt")
    else:
        with open(savepath+run_name+"_generalization_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
    
    env.close()
    #Plot rewards evolution
    if training_mode == True:
        # File path to save CSV
        csv_file_path = 'training_run_baseline/tensorboard_asp/csv/data_asp_run1.csv'
        # Writing list to CSV
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            ep = 1
            for item in total_rewards:
                writer.writerow([ep, item])
                ep += 1

        print("CSV file has been created successfully.")


        plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
        plt.plot(total_rewards)
        plt.show()


if __name__ == '__main__':
    run(asp=True, training_mode=training, pretrained=pretrained_weights)


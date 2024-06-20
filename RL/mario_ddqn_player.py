#Code modified from [on the Paperspace blog](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/).
#  https://github.com/Montyro/MarioSSRL
## Install the following if on a new instance, otherwise they'll ship with the container.
# !pip install nes-py==0.2.6
# !pip install gym-super-mario-bros
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y

import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from gym_super_mario_bros.actions import RIGHT_ONLY

import numpy as np
import cv2

from Environment.gym_wrappers import make_env
from DQNAgent import DQNAgent
from Configuration.config import config_player



config = config_player

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


def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]

#Shows current state (as seen in the emulator, not segmented)
def show_state(env, ep=0, info=""):
    cv2.imshow("Output!",env.render(mode='rgb_array')[:,:,::-1]) #Display using opencv
    cv2.waitKey(1)

def run(asp, pretrained):
   
    env = gym_super_mario_bros.make('SuperMarioBros-'+level+'-v0') #Load level
    env = make_env(env, input_type) # Wraps the environment so that frames are grayscale / segmented

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
            state_next, reward, terminal, info = env.step(int(action[0][0]))
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


#Code modified from [on the Paperspace blog](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/).
#  https://github.com/Montyro/MarioSSRL
## Install the following if on a new instance, otherwise they'll ship with the container.
# !pip install nes-py==0.2.6
# !pip install gym-super-mario-bros
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y

import csv
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
import pickle
from symbolic_components.Advisor import Advisor

from gym_super_mario_bros.actions import RIGHT_ONLY

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Configuration.config import config_asp, config

from Environment.gym_wrappers import make_env
from RL.DQNAgent import DQNAgent

import os

configuration = config_asp


### Run settings.
training = configuration['train']
vis = configuration['vis']
level = configuration['level']
use_tensorboard = configuration['tensorboard']

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter

run_name = configuration['run_name']

run_number = 1

if configuration['input_type'] == 'asp':
    input_type = 'asp'
else:
    input_type = 'rgb'

##Training settings:
if configuration['train'] ==  False:
    if configuration['inference_type'] == 'pure':
        max_exploration_rate = 0
        min_exploration_rate = 0
    else:
        max_exploration_rate = configuration['min_exp_r']
        min_exploration_rate = configuration['min_exp_r']
else:
    max_exploration_rate = configuration['max_exp_r']
    min_exploration_rate = configuration['max_exp_r']

epochs = configuration['epochs']

if configuration['backup_epochs'] > 0:
    backup_interval = configuration['backup_epochs']
else:
    backup_interval = -1

save_good_model = configuration['save_good_model']
#Model saving and loading
#Is there a directory for models? otherwise create it
dir_exist = os.path.exists(configuration['working_dir']) and os.path.isdir(configuration['working_dir'])
if not dir_exist:
    os.mkdir(configuration['working_dir'])
savepath = configuration['working_dir']+'/'

pretrained_weights = configuration['pretrained_weights']
pretrained_model_name = configuration['model_name']

#What to do with experience replay
load_exp_rep = configuration['load_experience_replay']
save_exp_rep = configuration['save_experience_replay']

def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]

#Shows current state (as seen in the emulator, not segmented)
def show_state(env, ep=0, info=""):
    cv2.imshow("Output!",env.render(mode='rgb_array')[:,:,::-1]) #Display using opencv
    cv2.waitKey(1)


def recover_asp_facts(tensor):
    tensor = tensor[0][5]
    facts = []
    for i in range (tensor.shape[0]):
        for j in range(tensor.shape[1]):
            if tensor[i][j] != 0:
                facts.append(f'cell({i},{j},{int(round(float(tensor[i][j]*255*6/255), 0))}).')
    facts = ' '.join(facts)
    return facts


def run(asp, training_mode, pretrained):

    on_ground = True
    previous_x = 0
    perform_no_op = False

    advisor = Advisor(config)
   
    env = gym_super_mario_bros.make('SuperMarioBros-'+level+'-v0') #Load level
    env = make_env(env, input_type) # Wraps the environment so that frames are 15x16 ASP frames

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
        tensorboard_writer = SummaryWriter(f'tensorboard_asp_advisor/run_{run_number}_labels')

    max_reward = 0
    current_counter = save_good_model

    #Each iteration is an episode (epoch)
    for ep_num in tqdm(range(epochs)):

        counter = 0

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
            action = int(action[0])
            # recover state_facts
            facts = recover_asp_facts(state)

            action, advice_given = advisor.advise(action, facts, on_ground)

            if perform_no_op:
                action = 0
                perform_no_op = False
                previous_x += 1

            #Increase step number
            steps += 1
            #Perform the action and advance to the next state
            state_next, reward, terminal, info = env.step(action)

            y_pos = info['y_pos']

            x_pos = info['x_pos']

            if x_pos == previous_x:
                counter += 1
                if counter == 5:
                    counter = 0
                    perform_no_op = True
            else:
                counter = 0

            previous_x = x_pos

            if y_pos == 79:
                on_ground = True
            else:
                on_ground = False

            action = torch.tensor([[action]])

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


                ######################### End of Model Backup Section #################################
                #Add state to experience replay "dataset"
                if not advice_given or terminal:
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

    else:
        with open(savepath+run_name+"_generalization_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
    
    env.close()
    #Plot rewards evolution
    if training_mode == True:
        # File path to save CSV
        # csv_file_path = 'training_run_baseline/tensorboard_asp/csv/data_asp_run1.csv'
        csv_file_path = f'/Users/maximvandecasteele/PycharmProjects/NeurASP/NeurASP/models/asp_ddqn_advisor/run{run_number}/data_asp_advisor_run{run_number}.csv'
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
    for i in range(5):
        run_number = i + 1
        savepath = f'/Users/maximvandecasteele/PycharmProjects/NeurASP/NeurASP/models/asp_ddqn_advisor/run{run_number}/'
        run(asp=True, training_mode=training, pretrained=pretrained_weights)


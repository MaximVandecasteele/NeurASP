import torch
import gym_super_mario_bros
import os
import csv
from utils import *
from DQNVanilla import Dqn_vanilla
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

device = 'cpu'
device_name = 'cpu'
if torch.backends.mps.is_available():
    mps_device = torch.device(device)
    print("Using mps device.")
    device = 'mps'
elif torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print("Using CUDA device:", device_name)
    device = 'cuda'
else:
    print("CUDA is not available")

config = {
    "device": device_name,
    # input dimensions of observation (64 objects of 5 characteristics, class, xmin, xmax, ymin, ymax)
    "observation_dim": (16, 16),
    # amount of frames to skip (skipframe)
    "skip": 4,
    # VecFrameStack
    "stack_size": 4,
}

ENV_NAME = 'SuperMarioBros-1-1-v0'
# if you want to see mario play
DISPLAY = False
CKPT_SAVE_INTERVAL = 1000
NUM_OF_EPISODES = 50_000

# 2. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
# env = JoypadSpace(env, RIGHT_ONLY)

# TODO folder creation
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)
# TODO observation space change
# 3. Apply the decorator chain
print(env.observation_space)
# TODO different wrappers
env = apply_wrappers(env, config)

# TODO different DQN
Dqn = Dqn_vanilla(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

env.reset()
# next_state, reward, done, trunc, info = env.step(action=0)

# TODO different log file
file_path = 'log/output_B1.csv'
os.makedirs(file_path, exist_ok=True)

# Appending to CSV
with open(file_path, mode='a+', newline='') as file:
    writer = csv.writer(file)

    for i in range(NUM_OF_EPISODES):
        print("Episode:", i)
        done = False
        state, _ = env.reset()
        total_reward = 0
        while not done:
            a = Dqn.choose_action(state)
            new_state, reward, done, truncated, info  = env.step(a)
            total_reward += reward

            Dqn.store_in_memory(state, a, reward, new_state, done)
            Dqn.learn()

            state = new_state

        data = [[i, total_reward, Dqn.loss_score.item(), Dqn.learn_step_counter, Dqn.epsilon, len(Dqn.replay_buffer)]]
        print("Total reward:", total_reward, "Loss:", Dqn.loss_score.item(), "Learn step counter:", Dqn.learn_step_counter,
              "Epsilon:", Dqn.epsilon, "Size of replay buffer:", len(Dqn.replay_buffer))

        for row in data:
            writer.writerow(row)

        print(f'Data has been appended to {file_path}')

        if (i + 1) % CKPT_SAVE_INTERVAL == 0:
            Dqn.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

        print("Total reward:", total_reward)

env.close()

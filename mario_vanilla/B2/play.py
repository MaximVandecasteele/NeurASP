import torch
import numpy as np
import gym_super_mario_bros
import os
from utils import *
from DQNAsp import Dqn_asp
from gym.vector.utils import spaces
from nes_py.wrappers import JoypadSpace
from wrappers import apply_ASP_wrappers
from mario_vanilla import Positioner
from mario_vanilla import (Detector)

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
    "detector_model_path": '../../Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
    "detector_label_path": '../../Object_detector/models/data.yaml',
    "positions_asp": '../asp/positions.lp',
    "show_asp": '../asp/show.lp',
}

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
# if you want to see mario play
DISPLAY = True
CKPT_SAVE_INTERVAL = 1000
NUM_OF_EPISODES = 50_000

architecture = 0

# 2. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
# env = JoypadSpace(env, RIGHT_ONLY)

model_path = os.path.join('B2', "models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)
# Create the object detector. This is a YOLO8 model
detector = Detector(config)
positioner = Positioner(config)
y, x = config["observation_dim"]
# env.observation_space = spaces.Box(low=-1, high=1024, shape=(config["observation_dim"],), dtype=np.float32)
env.observation_space = spaces.Box(low=0, high=10, shape=(y,x), dtype=np.int8)
print(env.observation_space)
# hack the observation space of the environment. We reduce to a single vector, but the environment is expecting
# a colored image. This can be overridden by setting the observation space manually
env = apply_ASP_wrappers(env, config, detector, positioner)

agent = Dqn_asp(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# folder_name = ""
# ckpt_name = ""
# agent.load_model(os.path.join("models", folder_name, ckpt_name))
agent.load_model('/Users/maximvandecasteele/PycharmProjects/NeurASP/mario_vanilla/models/B1/2024-03-25-15_56_15/model_50000_iter.pt')
agent.epsilon = 0.2
agent.eps_min = 0.0
agent.eps_decay = 0.0

env.reset()
# next_state, reward, done, trunc, info = env.step(action=0)

for i in range(NUM_OF_EPISODES):
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info  = env.step(a)
        total_reward += reward
        state = new_state

    # data = [[i, total_reward, agent.loss_score.item(), agent.learn_step_counter, agent.epsilon, len(agent.replay_buffer)]]
    # print("Total reward:", total_reward, "Loss:", agent.loss_score.item(), "Learn step counter:", agent.learn_step_counter, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer))

    # file_path = 'models/output_B1.csv'

    # if not os.path.isfile(file_path):
    #     with open(file_path, mode='w', newline='') as file:
    #         pass  # Create an empty file
    # # Appending to CSV
    # with open(file_path, mode='a+', newline='') as file:
    #     writer = csv.writer(file)
    #     for row in data:
    #         writer.writerow(row)

    # print(f'Data has been appended to {file_path}')

env.close()
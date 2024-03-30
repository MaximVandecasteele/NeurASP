import torch
import gym_super_mario_bros
import os
import re
from utils import *
import csv
import numpy as np

from nes_py.wrappers import JoypadSpace
from DQN import Dqn
from gym.vector.utils import spaces
from wrappers import apply_wrappers, apply_ASP_wrappers
from mario_vanilla.symbolic_components.positioner import Positioner
from mario_vanilla.symbolic_components.detector import Detector

class Evaluator(object):
    def __init__(self):

        self.ENV_NAME = 'SuperMarioBros-1-1-v0'

        self.device = 'cpu'
        device_name = 'cpu'
        if torch.backends.mps.is_available():
            mps_device = torch.device(self.device)
            print("Using mps device.")
            device = 'mps'
        elif torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print("Using CUDA device:", device_name)
            device = 'cuda'
        else:
            print("CUDA is not available")

        self.config = {
            "device": device_name,
            # input dimensions of observation (64 objects of 5 characteristics, class, xmin, xmax, ymin, ymax)
            "observation_dim": (15, 16),
            # amount of frames to skip
            "skip": 4,
            # VecFrameStack
            "stack_size": 4,
            "detector_model_path": '../Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
            "detector_label_path": '../Object_detector/models/data.yaml',
            "positions_asp": './asp/positions.lp',
            "show_asp": './asp/show.lp',
        }

        self.detector = Detector(self.config)
        self. positioner = Positioner(self.config)

    def init_environment(self, display, asp):
        # Create the base environment
        env = gym_super_mario_bros.make(self.ENV_NAME, render_mode='human' if display else 'rgb',
                                        apply_api_compatibility=True)
        if asp:
            # hack the observation space of the environment.
            y, x = self.config["observation_dim"]
            env.observation_space = spaces.Box(low=0, high=10, shape=(y, x), dtype=np.int8)
            env = apply_ASP_wrappers(env, self.config, self.detector, self.positioner)
            print(env.observation_space)
        else:
            env = apply_wrappers(env, self.config)
            print(env.observation_space)
        return env

    @staticmethod
    def build_dqn(input_dim, action_space, asp):
        return Dqn(input_dims=input_dim, num_actions=action_space, eps_min=0.0, eps_decay=0.0, asp=asp)


    def evaluate(self, num_episodes, exp_name, env, dqn, eval_path):
        os.makedirs(eval_path, exist_ok=True)
        eval_file = os.path.join(eval_path, f"performance_{exp_name}.csv")

        if not os.path.isfile(eval_file):
            with open(eval_file, mode='w', newline='') as file:
                pass  # Create an empty file
        # Appending to CSV
        with open(eval_file, mode='a+', newline='') as file:
            writer = csv.writer(file)

            # Specify the directory path
            directory_path = f'{exp_name}/models'

            # Iterate through the contents of the directory
            for filename in os.listdir(directory_path):
                if filename != '.DS_Store':
                    match = re.search(r'\d+', filename)
                    if match:
                        model_number = int(match.group())
                        print(f"model: {model_number}")
                    # TODO iets met seeds, zorgen voor variatie bij trainen?
                    # Join the directory path with the filename to get the absolute path
                    model = os.path.join(directory_path, filename)
                    dqn.load_model(model)
                    # TODO adapt epsilon Exploitation-Only Testing vs Exploration-Exploitation Testing
                    dqn.epsilon = 0.05
                    # hoe ga ik dit doen met die epsilon?
                    for i in range(num_episodes):
                        print(f"model: {model_number}, episode: {i}")
                        done = False
                        state, _ = env.reset()
                        total_reward = 0
                        while not done:
                            a = dqn.choose_action(state)
                            new_state, reward, done, truncated, info = env.step(a)
                            total_reward += reward
                            state = new_state
                        # write data to csv
                        data = [[model_number, i, total_reward, dqn.epsilon]]
                        for row in data:
                            writer.writerow(row)

        env.close()







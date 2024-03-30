import torch
import numpy as np
import gym_super_mario_bros
import os
import csv
from utils import *
from B1.DQNVanilla import Dqn_vanilla
from B2.DQNAsp import Dqn_asp

from gym.vector.utils import spaces
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers, apply_ASP_wrappers
from mario_vanilla.symbolic_components.positioner import Positioner
from mario_vanilla.symbolic_components.detector import Detector

class Trainer(object):
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
        if asp:
            return Dqn_asp(input_dims=input_dim, num_actions=action_space)
        else:
            return Dqn_vanilla(input_dims=input_dim, num_actions=action_space)

    @staticmethod
    def train(num_episodes, save_interval, exp_name, env, dqn, model_path, log_path):
        env.reset()
        # model folder creation
        # model_path = os.path.join("models", self.exp_name, get_current_date_time_string())
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        log_file = os.path.join(log_path, f"output_{exp_name}.csv")

        if not os.path.isfile(log_file):
            with open(log_file, mode='w', newline='') as file:
                pass  # Create an empty file
        # Appending to CSV
        with open(log_file, mode='a+', newline='') as file:
            writer = csv.writer(file)

            for i in range(num_episodes):
                print("Episode:", i)
                done = False
                state, _ = env.reset()
                total_reward = 0
                while not done:
                    a = dqn.choose_action(state)
                    new_state, reward, done, truncated, info = env.step(a)
                    total_reward += reward

                    dqn.store_in_memory(state, a, reward, new_state, done)
                    dqn.learn()

                    state = new_state

                data = [[i, total_reward, dqn.loss_score.item(), dqn.learn_step_counter, dqn.epsilon, len(dqn.replay_buffer)]]
                print("Total reward:", total_reward, "Loss:", dqn.loss_score.item(), "Learn step counter:", dqn.learn_step_counter,
                      "Epsilon:", dqn.epsilon, "Size of replay buffer:", len(dqn.replay_buffer))

                for row in data:
                    writer.writerow(row)

                if (i + 1) % save_interval == 0:
                    dqn.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

        env.close()

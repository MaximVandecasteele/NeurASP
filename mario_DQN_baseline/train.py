# code heavily inspired by Nicholas Renotte's tutorial
# YouTube: https://www.youtube.com/watch?v=2eeYqJ0uBKE
# GitHub: https://github.com/nicknochnack/MarioRL

import gym_super_mario_bros
import numpy as np
import torch
from gym.vector.utils import spaces
from nes_py.wrappers import JoypadSpace
from stable_baselines3_master.stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3_master.stable_baselines3.common import env_checker

from mario_DQN_baseline.callback import CheckpointCallback, IntervalCallback, EpisodeCallback
from mario_DQN_baseline.symbolic_components.detector import Detector
from mario_DQN_baseline.wrappers import apply_wrappers, apply_ASP_wrappers
from mario_DQN_baseline.symbolic_components.positioner import Positioner
# Import DQN for algos
from stable_baselines3_master.stable_baselines3 import DQN

from mario_DQN_baseline.our_logging import our_logging

LOG_TIMING = True
our_logging.initialize(LOG_TIMING)

# nes_py bugfix
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

ENV_NAME = 'SuperMarioBros-1-1-v0'
# if you want to see mario play
DISPLAY = True
CHECKPOINT_FREQUENCY = 200000
TOTAL_TIME_STEPS = 8000000
CHECKPOINT_DIR = 'train/'
TENSORBOARD_LOG_DIR = 'logs/tensorboard/'

architecture = 0

device = 'cpu'
device_name = 'cpu'
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device = 'cuda'

config = {
    "device": device_name,
    # input dimensions of observation (64 objects of 5 characteristics, class, xmin, xmax, ymin, ymax)
    "observation_dim": (15, 16, 1),
    # amount of frames to skip (skipframe)
    "skip": 4,
    # VecFrameStack
    "stack_size": 4,
    # "learning_rate": 0.000001,
    # also 'MlpPolicy (Zorgen voor multidimensionele input in geval van CNN)
    "rl_policy": 'CnnPolicy',
    # "rl_policy": 'MlpPolicy',
    "detector_model_path": '../Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
    "detector_label_path": '../Object_detector/models/data.yaml',
    "positions_asp": './asp/positions.lp',
    "show_asp": './asp/show.lp',
}

# Setup game
# 1. Create the object detector. This is a YOLO8 model
detector = Detector(config)
positioner = Positioner(config)

# 2. Create the base environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)

if architecture == 0:
    # 3. Apply the decorator chain
    print(env.observation_space)
    env = apply_wrappers(env, config)
elif architecture == 1:
    y, x, chann = config["observation_dim"]
    # env.observation_space = spaces.Box(low=-1, high=1024, shape=(config["observation_dim"],), dtype=np.float32)
    env.observation_space = spaces.Box(low=0, high=255, shape=(y,x, chann), dtype=np.int8)



    print(env.observation_space)
    # hack the observation space of the environment. We reduce to a single vector, but the environment is expecting
    # a colored image. This can be overridden by setting the observation space manually
    env = apply_ASP_wrappers(env, config, detector, positioner)


# 4. Start the game
state = env.reset()

# Setup model saving callback and pass the configuration, so we know the exact configuration belonging to the logs

# save model at certain intervals
checkpointCallback = CheckpointCallback(check_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINT_DIR, model_name='B1', config=config)
intervalCallback = IntervalCallback(check_freq=1)
episodeCallback = EpisodeCallback()

# This is the AI model
model = DQN(
    config["rl_policy"],
    env,
    verbose=1,
    train_freq=1,
    # How many gradient steps to take after each rollout
    gradient_steps=1,
    learning_rate=0.00025,
    gamma=0.9,
    exploration_initial_eps=1.0,
    # fraction of entire training period over which the exploration rate is reduced
    exploration_fraction=0.9,
    # final value of random action probability
    exploration_final_eps=0.1,
    # update the target network every target_update_interval environment steps.
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=100000,
    batch_size=32,
    policy_kwargs=dict(net_arch=[512, 512]),
    tensorboard_log=TENSORBOARD_LOG_DIR,
    device=device,
    seed=1,
)

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=[checkpointCallback,intervalCallback,episodeCallback], tb_log_name='vanilla_DQN_B1')

# https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
# mean_reward, std_reward = evaluate_policy(
#     model,
#     model.get_env(),
#     deterministic=True,
#     n_eval_episodes=20,
# )

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

print("Training done")








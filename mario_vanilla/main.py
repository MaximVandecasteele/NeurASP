from mario_vanilla.trainer import Trainer
from mario_vanilla.evaluator import Evaluator
import numpy as np
from utils import *
import os
import torch

DISPLAY = True
UBUNTU = False
ASP = True
MLP = True

if torch.backends.mps.is_available():
    # mps_device = torch.device(self.device)
    print("Using mps device.")
    device = 'mps'
elif torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(1)
    print("Using CUDA device:", device_name)
    device = 'cuda:1'
else:
    print("CUDA is not available")
    device = 'cpu'


if UBUNTU:
    if MLP:
        config = {
            "device": device,
            # input dimensions of observation (64 objects of 5 characteristics, class, xmin, xmax, ymin, ymax)
            "observation_dim": (5, 6),
            # TODO: remove
            "cnn_input_dim": (7, 15, 16),
            # amount of frames to skip
            "skip": 4,
            # VecFrameStack
            "stack_size": 4,
            "detector_model_path": '/home/stefaan/local/python/NeurASP/Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
            "detector_label_path": '/home/stefaan/local/python/NeurASP/Object_detector/models/data.yaml',
            "positions_asp": '/home/stefaan/local/python/NeurASP/mario_vanilla/asp/positions.lp',
            "show_asp": '/home/stefaan/local/python/NeurASP/mario_vanilla/asp/show.lp',
        }
    else:
        config = {
            "device": device,
            # input dimensions of observation (64 objects of 5 characteristics, class, xmin, xmax, ymin, ymax)
            "observation_dim": (15, 16),
            # TODO: remove
            "cnn_input_dim": (7, 15, 16),
            # amount of frames to skip
            "skip": 4,
            # VecFrameStack
            "stack_size": 4,
            "detector_model_path": '/home/stefaan/local/python/NeurASP/Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
            "detector_label_path": '/home/stefaan/local/python/NeurASP/Object_detector/models/data.yaml',
            "positions_asp": '/home/stefaan/local/python/NeurASP/mario_vanilla/asp/positions.lp',
            "show_asp": '/home/stefaan/local/python/NeurASP/mario_vanilla/asp/show.lp',
        }
else:
    if MLP:
        config = {
            "device": device,
            # input dimensions of observation (64 objects of 5 characteristics, class, xmin, xmax, ymin, ymax)
            "observation_dim": (5, 6),
            # TODO: remove
            "cnn_input_dim": (7, 15, 16),
            # amount of frames to skip
            "skip": 4,
            # VecFrameStack
            "stack_size": 4,
            "detector_model_path": '/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
            "detector_label_path": '/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/models/data.yaml',
            "positions_asp": '/Users/maximvandecasteele/PycharmProjects/NeurASP/mario_vanilla/asp/positions_lens.lp',
            "show_asp": '/Users/maximvandecasteele/PycharmProjects/NeurASP/mario_vanilla/asp/show.lp',
        }
    else:
        config = {
            "device": device,
            # input dimensions of observation (64 objects of 5 characteristics, class, xmin, xmax, ymin, ymax)
            "observation_dim": (15, 16),
            # TODO: remove
            "cnn_input_dim": (7, 15, 16),
            # amount of frames to skip
            "skip": 4,
            # VecFrameStack
            "stack_size": 4,
            "detector_model_path": '/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/models/YOLOv8-Mario-lvl1-3/weights/best.pt',
            "detector_label_path": '/Users/maximvandecasteele/PycharmProjects/NeurASP/Object_detector/models/data.yaml',
            "positions_asp": '/Users/maximvandecasteele/PycharmProjects/NeurASP/mario_vanilla/asp/positions.lp',
            "show_asp": '/Users/maximvandecasteele/PycharmProjects/NeurASP/mario_vanilla/asp/show.lp',
        }

# set seed for randomness, also works for pytorch code
np.random.seed(1)

# first train B2
trainer = Trainer(config)
env = trainer.init_environment(display=DISPLAY, asp=ASP)
dqn = trainer.build_dqn(input_dim=env.observation_space.shape, action_space=env.action_space.n, asp=ASP)


exp_name = 'B2_mlp_test'
model_path = os.path.join(exp_name, "models")
log_path = os.path.join(exp_name, "log")
# log_path = 'output_B2.csv'

trainer.train(num_episodes=50000, save_interval=1000, exp_name=exp_name, env=env, dqn=dqn, model_path=model_path, log_path=log_path)

# # empty the memory
# trainer = None

#  evaluate performance B1
# evaluator = Evaluator()
# asp = False
#
# env = evaluator.init_environment(display=False, asp=asp)
# dqn = evaluator.build_dqn(input_dim=env.observation_space.shape, action_space=env.action_space.n, asp=asp)
#
# exp_name = 'B1'
# eval_path = os.path.join(exp_name, "eval")
# evaluator.evaluate(num_episodes=100, exp_name=exp_name, env=env, dqn=dqn, eval_path=eval_path)


# evaluate performance B2
# asp = True
# evaluator = Evaluator()
#
#
# env = evaluator.init_environment(display=False, asp=asp)
# dqn = evaluator.build_dqn(input_dim=env.observation_space.shape, action_space=env.action_space.n, asp=asp)
#
# exp_name = 'B2_one_hot'
# eval_path = os.path.join(exp_name, "eval")
# evaluator.evaluate(num_episodes=100, exp_name=exp_name, env=env, dqn=dqn, eval_path=eval_path)

# evaluate generalization B1




# evaluate generalization B2



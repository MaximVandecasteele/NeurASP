from mario_vanilla.trainer import Trainer
from utils import *
import os

trainer = Trainer()

env = trainer.init_environment(display=True, asp=True)
dqn = trainer.build_dqn(input_dim=env.observation_space.shape, action_space=env.action_space.n, asp=True)

exp_name = 'B2'
model_path = os.path.join(exp_name, "models", get_current_date_time_string())
log_path = os.path.join(exp_name, "log")
# log_path = 'output_B2.csv'

trainer.train(num_episodes=50000, save_interval=1000, exp_name=exp_name, env=env, dqn=dqn, model_path=model_path, log_path=log_path)

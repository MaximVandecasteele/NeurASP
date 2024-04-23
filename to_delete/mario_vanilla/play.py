import torch

from mario_vanilla.trainer import Trainer

from nes_py.wrappers import JoypadSpace

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

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
# if you want to see mario play
DISPLAY = True

NUM_OF_EPISODES = 50

# 2. Create the base environment
asp = False
# 3. Apply the decorator chain

trainer = Trainer(config)

env = trainer.init_environment(display=True, asp=asp)
agent = trainer.build_dqn(input_dim=env.observation_space.shape, action_space=env.action_space.n,asp=asp)


folder_name = ""
ckpt_name = ""
# agent.load_model(os.path.join("models", folder_name, ckpt_name))
agent.load_model('/Users/maximvandecasteele/PycharmProjects/NeurASP/mario_vanilla/trained_models/B1/models/model_50000_iter.pt')
agent.epsilon = 0.15
agent.eps_min = 0.0
agent.eps_decay = 0.0



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
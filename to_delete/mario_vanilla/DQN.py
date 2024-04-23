import torch
import numpy as np
from mario_vanilla.DQN_architectures.DQN_vanilla_nn import Dqn_vanilla_nn
from mario_vanilla.DQN_architectures.DQN_asp_nn_one_hot import Dqn_asp_nn_one_hot
from mario_vanilla.DQN_architectures.DQN_asp_mlp_focus import Dqn_asp_mlp_focus

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class Dqn:
    def __init__(self, 
                 input_dims, 
                 num_actions,
                 asp,
                 lr=0.00025, 
                 gamma=0.9, 
                 epsilon=1.0,
                 # epsilon=0.4,
                 eps_decay=0.99999975, 
                 eps_min=0.1, 
                 replay_buffer_capacity=80_000,
                 batch_size=32, 
                 sync_network_rate=10000,
                 seed=1):
        self.asp = asp
        self.num_actions = num_actions
        self.learn_step_counter = 0
        self.loss_score = 0
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate
        self.seed = seed

        np.random.seed(self.seed)

        # Networks
        if asp:
            # TODO REMOVE
            # self.online_network = Dqn_asp_nn(input_dims, num_actions)
            # self.target_network = Dqn_asp_nn(input_dims, num_actions, freeze=True)
            self.online_network = Dqn_asp_nn_one_hot(input_dims, num_actions)
            self.target_network = Dqn_asp_nn_one_hot(input_dims, num_actions, freeze=True)
            # self.online_network = Dqn_asp_mlp_focus(input_dims, num_actions)
            # self.target_network = Dqn_asp_mlp_focus(input_dims, num_actions, freeze=True)
        else:
            self.online_network = Dqn_vanilla_nn(input_dims, num_actions)
            self.target_network = Dqn_vanilla_nn(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.SmoothL1Loss() # Feel free to try this loss function instead!

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        # Passing in a list of numpy arrays is slower than creating a tensor from a numpy array
        # Hence the `np.array(observation)` instead of `observation`
        # observation is a LIST of numpy arrays because of the LazyFrame wrapper
        # Unqueeze adds a dimension to the tensor, which represents the batch dimension
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        # Grabbing the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))
        
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        # TODO set back to removed map_location
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))
        # self.online_network.load_state_dict(torch.load(path,map_location='mps'))
        # self.target_network.load_state_dict(torch.load(path,map_location='mps'))

    def change_seed(self, seed):
        np.random.seed(seed)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()
        
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(states) # Shape is (batch_size, n_actions)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        # Max returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        # The rewards of any future states don't matter if the current state is a terminal state
        # If done is true, then 1 - done is 0, so the part after the plus sign (representing the future rewards) is 0
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        self.loss_score = self.loss(predicted_q_values, target_q_values)

        self.loss_score.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()


        


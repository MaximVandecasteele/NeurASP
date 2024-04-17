import random
import torch
from DQN_network_vanilla import DQNSolver
from DQN_network_asp import DQNSolver_asp
import pickle
import torch.nn as nn



#### Definition of the DQN Agent.
class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, pretrained, savepath, load_exp_rep,
                 pretrained_model_name, asp):

        if torch.backends.mps.is_available():
            # mps_device = torch.device(self.device)
            print("Using mps device.")
            self.device = 'mps'
        elif torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(1)
            print("Using CUDA device:", device_name)
            self.device = 'cuda:1'
        else:
            print("CUDA is not available")
            self.device = 'cpu'

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained

        if asp:
            self.local_net = DQNSolver_asp(state_space, action_space).to(self.device)
            self.target_net = DQNSolver_asp(state_space, action_space).to(self.device)
        else:
            self.local_net = DQNSolver(state_space, action_space).to(self.device)
            self.target_net = DQNSolver(state_space, action_space).to(self.device)

        if self.pretrained:
            self.local_net.load_state_dict(
                torch.load(savepath + pretrained_model_name + "dq1.pt", map_location=torch.device(self.device)))
            self.target_net.load_state_dict(
                torch.load(savepath + pretrained_model_name + "dq2.pt", map_location=torch.device(self.device)))

        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.copy = 1000  # Copy the local model weights into the target network every 1000 steps
        self.step = 0

        # Reserve memory for the experience replay "dataset"
        self.max_memory_size = max_memory_size

        if load_exp_rep:
            self.STATE_MEM = torch.load(savepath + "STATE_MEM.pt")
            self.ACTION_MEM = torch.load(savepath + "ACTION_MEM.pt")
            self.REWARD_MEM = torch.load(savepath + "REWARD_MEM.pt")
            self.STATE2_MEM = torch.load(savepath + "STATE2_MEM.pt")
            self.DONE_MEM = torch.load(savepath + "DONE_MEM.pt")

            if True:  # If you get errors loading ending positions or num in queue just change this to False
                with open(savepath + "ending_position.pkl", 'rb') as f:
                    self.ending_position = pickle.load(f)
                with open(savepath + "num_in_queue.pkl", 'rb') as f:
                    self.num_in_queue = pickle.load(f)
            else:
                self.ending_position = 0
                self.num_in_queue = 0
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Set up agent learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done):  # Store "remembrance" on experience replay
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def recall(self):
        # Randomly sample 'batch size' experiences from the experience replay
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]

        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        # Epsilon-greedy action
        self.step += 1

        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])

        # Local net is used for the policy
        logits = self.local_net(state.to(self.device))

        action = torch.argmax(logits).unsqueeze(0).unsqueeze(0).cpu()

        return action

    def copy_model(self):
        # Copy local net weights into target net
        self.target_net.load_state_dict(self.local_net.state_dict())

    def experience_replay(self):
        if self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()

        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
        target = REWARD + torch.mul((self.gamma *
                                     self.target_net(STATE2).max(1).values.unsqueeze(1)),
                                    1 - DONE)
        current = self.local_net(STATE).gather(1, ACTION.long())  # Local net approximation of Q-value

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)





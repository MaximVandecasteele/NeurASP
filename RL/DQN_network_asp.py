import torch.nn as nn
import torch
import numpy as np


#### Definition of the DQN model
class DQNSolver_asp(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSolver_asp, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=2, stride=1),
            # nn.ReLU()

            # #Dit was een ok versie, net iets slechter dan vanilla
            # nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU()

            # # Deze begon heel snel, maar vlakte dan enorm af op 1250
            # nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=4, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU()

            # # Deze begon heel snel, maar vlakte dan enorm af op 1250
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.gradients = None

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)

        conv_out.register_hook(self.activations_hook)

        return self.fc(conv_out.view(x.size()[0], -1))

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.conv(x)
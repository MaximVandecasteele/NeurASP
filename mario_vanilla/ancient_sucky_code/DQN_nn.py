import torch
from torch import nn
import numpy as np

class Dqn_nn(nn.Module):
    def __init__(self, input_shape, n_actions, asp, freeze=False):
        super().__init__()
        # Convolutional layers
        if asp:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                # nn.Conv2d(64, 64, kernel_size=3, stride=1),
                # nn.ReLU(),
            )
        else:
            self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            )


        conv_out_size = self._get_conv_out(input_shape)

        if asp:
            # Linear layers
            self.network = nn.Sequential(
                self.conv_layers,
                nn.Flatten(),
                nn.Linear(conv_out_size, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )
        else:
            # Linear layers
            self.network = nn.Sequential(
                self.conv_layers,
                nn.Flatten(),
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )

        if freeze:
            self._freeze()
        
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.to(self.device)


    def forward(self, x):
        return self.network(x)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(o.size()))
    
    def _freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False
    
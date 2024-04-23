import torch
from torch import nn
import numpy as np

class Dqn_asp_mlp_focus(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        # Linear layers
        self.network = nn.Sequential(

                nn.Linear(210, 100),
                nn.ReLU(),
                nn.Linear(100, n_actions)
        )

        if freeze:
            self._freeze()
        
        # self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            print("Using mps device.")
            self.device = 'mps'
        elif torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(1)
            print("Using CUDA device:", device_name)
            self.device = torch.device("cuda:1")
        else:
            print("CUDA is not available")
            self.device = 'cpu'

        # self.device = torch.device("cuda:1")
        self.to(self.device)

    def forward(self, x):
        return self.network(x)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(*shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(o.size()))
    
    def _freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False
    
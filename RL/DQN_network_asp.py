import torch.nn as nn
import torch
import numpy as np


#### Definition of the DQN model
class DQNSolver_asp(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSolver_asp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            # only for neurasp purposes, otherwise comment out the softmax
            # nn.Softmax(dim=0)
        )

        self.gradients = None

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)

        conv_out.register_hook(self.activations_hook)
        reshape = conv_out.view(x.size()[0], -1)
        # reshape = conv_out.view(3584)
        return self.fc(reshape)

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.conv(x)


def testNN(model, tensors_test, symbols_test, device):
    """
    Return a real number "accuracy" in [0,100] which counts 1 for each data instance;
           a real number "singleAccuracy" in [0,100] which counts 1 for each number in the label
    @param model: a PyTorch model whose accuracy is to be checked
    @oaram testLoader: a PyTorch dataLoader object, including (input, output) pairs for model
    """
    # set up testing mode
    model.eval()

    # check if total prediction is correct
    correct = 0
    total = 0
    # check if each single prediction is correct
    singleCorrect = 0
    singleTotal = 0

    indices = len(tensors_test)
    for index in range(indices):
        tensor = tensors_test[index]['state'].to(device)
        symbols = symbols_test[index]

        output = model(tensor)
        pred = output.argmax(dim=-1)

        correct += 1 if pred in symbols else False
        total += 1

    accuracy = 100. * correct / total

    return accuracy
"""

We need a neural network that 
1. inputs a board, 2d numpy array
2. outputs a value and probability distribution, i.e. 2 heads


"""

import torch
import torch.nn as nn
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_channels = 1 
        output_features = 32
        self.conv1 = nn.Conv2d(
            input_channels, output_features,
            kernel_size=2, stride=1, padding='same'
        )

        hidden_units = 100
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features = 4*4*output_features, 
                            out_features = hidden_units)

        self.output_layers = Split(
            nn.Sequential(
                nn.Linear(hidden_units, 4),
                nn.Softmax()
            ),
            nn.Sequential(
                nn.Linear(hidden_units, 1),
                nn.ReLU()
            ),        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x) 
        out = self.output_layers(x)
        return out


class Split(nn.Module):
    def __init__(self, *modules: torch.nn.Module):
        super().__init__()
        self.modules = modules

    def forward(self, inputs):
        return [module(inputs) for module in self.modules]   

if __name__ == '__main__':
    net = Net()   
    print(net) 
    summary(net, (1, 4, 4))

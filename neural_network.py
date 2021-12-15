"""

We need a neural network that 
1. inputs a board, 2d numpy array
2. outputs a value and probability distribution, i.e. 2 heads


"""
import numpy as np
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

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    def forward(self, output, labels):
        #print(output, labels)
        _, label= labels[0].clone().detach().max(dim=0)
        loss_p = nn.CrossEntropyLoss()(output[0], torch.tensor([label]))
        loss_v = nn.MSELoss()(output[1], labels[1].unsqueeze(0).float())
        return sum([loss_p,loss_v]).float()
    



class Split(nn.Module):
    def __init__(self, *modules: torch.nn.Module):
        super().__init__()
        self.modules = modules

    def forward(self, inputs):
        return [module(inputs) for module in self.modules]   


def get_dummy():
    dummy_data = np.random.rand(10, 4, 4).astype(np.float32)
    dummy_labels = []
    for i in range(10):
        if i %2:
            dummy_labels.append(
            [np.array([0, 0, 0, 1], dtype=np.float32), np.array([3], dtype=np.float32)]
        )
        else:
            dummy_labels.append(
            [np.array([0, 0, 1, 0], dtype=np.float32), np.array([3], dtype=np.float32)]
        )
    return dummy_data, dummy_labels

def train_network(network, data, target, epochs_count):
    network.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.from_numpy(data).float().to(device)
    torch_data = torch.unsqueeze(data, 1)
    print(torch_data.size())
    torch_target = [
        [torch.from_numpy(elem[0]).squeeze().float().to(device), torch.tensor([elem[1]], dtype=float).to(device)] for elem in target
    ]

    criterion = CustomLoss()

    optimizer = torch.optim.Adam(params = network.parameters(), lr=0.1)

    for _ in range(epochs_count):
        for sample, label in zip(torch_data, torch_target):
            sample = sample.unsqueeze(0)
            output = network(sample)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        pass


    print("completed training")

if __name__ == '__main__':
    net = Net()   
    print(net) 
    summary(net, (1, 4, 4))
    data, target = get_dummy()
    train_network(net, data, target)

"""

We need a neural network that 
1. inputs a board, 2d numpy array
2. outputs a value and probability distribution, i.e. 2 heads


"""
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm

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

        self.fc2 = nn.Linear(in_features = hidden_units, 
                            out_features = hidden_units)
        self.relu = nn.ReLU()
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
        x = self.relu(x)
    
        x = self.flatten(x)
        x = self.fc1(x) 
        #x = self.relu(x)
        #x = self.fc2(x) 
        x = self.relu(x)

        out = self.output_layers(x)
        return out

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    def forward(self, output, labels):
        output_p = output[:, :4]
        output_v = output[:, 4]
        label_p = labels[:, :4]
        label_v = labels[:, 4]

        loss_p = nn.CrossEntropyLoss()(output_p, label_p)
        loss_v = nn.MSELoss()(output_v, label_v/10)
        return sum([loss_p,loss_v]).float()
    



class Split(nn.Module):
    def __init__(self, *modules: torch.nn.Module):
        super().__init__()
        self.modules = modules

    def forward(self, inputs):
        return torch.hstack((self.modules[0](inputs), self.modules[1](inputs)))


def get_dummy():
    dummy_data = np.random.rand(100, 4, 4).astype(np.float32)
    dummy_labels = []
    for i in range(100):
        if i %2:
            dummy_labels.append(
            np.array([0, 0, 0, 1, 3], dtype=np.float32)
        )
        else:
            dummy_labels.append(
            np.array([0, 0, 1, 0, 3], dtype=np.float32)
        )
    print("data", dummy_data.shape)
    print("labels", np.array(dummy_labels).shape)
    return dummy_data, np.array(dummy_labels, dtype=np.float32)

def train_network(network, data, target, epochs_count=10):
    network.train()
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
 

    criterion = CustomLoss()

    optimizer = torch.optim.Adam(params = network.parameters(), lr=0.001)
    batch_size = min(data.shape[0], 100)
    for i in tqdm(range(epochs_count)):
        for idx in range(data.shape[0]//batch_size):
            optimizer.zero_grad()

            samples = torch.from_numpy(data[idx*batch_size: (idx+1)*batch_size]).float().to(device).unsqueeze(1)
            targets = torch.from_numpy(target[idx*batch_size: (idx+1)*batch_size]).float().to(device)
            output = network(samples)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
    print(loss)
        


    print("completed training")

if __name__ == '__main__':
    net = Net()   
    print(net) 
    summary(net, (1, 4, 4))
    data, target = get_dummy()
    train_network(net, data, target)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, num_outputs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_outputs)

        self.LReLU = nn.LeakyReLU(0.01)

        self.drop_embs = nn.Dropout(p=0.2)  # 0.2 dropout
        self.drop_hidden = nn.Dropout(p=0.5)  # 0.5 dropout

    def forward(self, x):
        # input layer
        x = self.drop_embs(x)

        # first hidden
        x = self.fc1(x)
        x = self.LReLU(x)
        x = self.drop_hidden(x)

        # second hidden
        x = self.fc2(x)
        x = self.LReLU(x)
        x = self.drop_hidden(x)

        # output layer
        x = self.fc3(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)

        self.fc1 = nn.Linear(912, 400)
        self.fc2 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, R):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x)
        R = torch.flatten(R)
        hh = torch.cat((x, R), dim=0)
        hh = F.relu(self.fc1(hh))
        hh = F.relu(self.fc2(hh))
        out = self.fc3(hh)
        return out


import torch.nn as nn
import torch
import torch.nn.functional as F

class Mil_classifier(nn.Module) :
    def __init__(self):
        # signal = [[],[],[]] = 1
        super(Mil_classifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, 100, stride=40)
        self.bn1 = nn.BatchNorm1d(20)
        self.pool1 = nn.MaxPool1d(15, 6)

        self.conv2 = nn.Conv1d(20, 40, 5)
        self.bn2 = nn.BatchNorm1d(40)
        self.pool2 = nn.MaxPool1d(5, 2)

        self.conv3 = nn.Conv1d(40, 20, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(20)
        self.pool3 = nn.MaxPool1d(3, 2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10,1)

    def forward(self, x : torch.tensor):
        # Input: [B, N, L]
        B, N, L = x.shape

        # Treat every segment as an individual signal.
        # [B, N, L] -> [B*N, 1, L]
        x = x.reshape(B * N, 1, L)

        x = self.bn1(self.conv1(x))
        x = self.pool1(F.relu(x))

        x = self.bn2(self.conv2(x))
        x = self.pool2(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool3(F.relu(x))

        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))

        
        x = self.fc2(x)
        x = x.reshape(B, N)

        return x

        # x = x.max(dim=0).values
        # return x
import torch.nn as nn
import torch
import torch.nn.functional as F

class Mil_classifier_128(nn.Module) :
    def __init__(self):
        # signal = [[],[],[]] = 1
        super(Mil_classifier_128, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 40, stride=10) #40, s=10, 299 == 597 * 128
        self.bn1 = nn.GroupNorm(4, 128)
        self.pool1 = nn.MaxPool1d(10, 5) # 10,5 = 582
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv1d(128, 50, 5) #30, 5 = 111
        self.bn2 = nn.GroupNorm(5, 50)
        self.pool2 = nn.MaxPool1d(5, 2) # 53
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv1d(50, 128, 10, 5) #10,5 : 39
        self.bn3 = nn.GroupNorm(4, 128)
        # self.pool3 = nn.MaxPool1d(3, 2)
        self.dropout3 = nn.Dropout(p=0.3)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 2*128)
        self.fc2 = nn.Linear(2*128,1)

    def forward(self, x : torch.tensor):
        # Input: [B, N, L]
        B, N, L = x.shape
       
        # Treat every segment as an individual signal.
        # [B, N, L] -> [B*N, 1, L]
        x = x.reshape(B * N, 1, L)
       
        x = self.bn1(self.conv1(x))
       
        x = self.pool1(self.dropout1(F.relu(x)))

        x = self.bn2((self.conv2(x)))
        x = self.pool2(self.dropout2(F.relu(x)))

        x = self.bn3(self.conv3(x))
        x = self.dropout3(F.relu(x))

        x = self.gap(x).squeeze(-1)

        x = x.mean(dim = 0)
        x = F.relu(self.fc1(x))

        
        x = self.fc2(x)

        return x

        # x = x.max(dim=0).values
        # return x
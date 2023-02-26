import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ultimus(nn.Module):
    def __init__(self):
        super(Ultimus, self).__init__()
        self.fc1_k = nn.Linear(48, 8)
        self.fc2_q = nn.Linear(48, 8)
        self.fc3_v = nn.Linear(48, 8)
        self.out = nn.Linear(8, 48)
       
    def forward(self, x):
        orig = x
        k = self.fc1_k(x)
        q = self.fc2_q(x)
        v = self.fc3_v(x)
        am = F.softmax(torch.div(torch.multiply(q, k),(8**0.5)))
        z = torch.multiply(v, am)
        out = self.out(z)
        out = torch.add(out,orig)
        return out


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.conv1 = nn.Sequential(
          nn.Conv2d(3, 16, 3, padding=1),
          nn.ReLU()
        )
        self.conv2 = nn.Sequential(
          nn.Conv2d(16, 32, 3, padding=1),
          nn.ReLU()
        )
        self.conv3 = nn.Sequential(
          nn.Conv2d(32, 48, 3, padding=1),
          nn.ReLU()
        )
        self.ult1 = Ultimus()
        self.ult2 = Ultimus()
        self.ult3 = Ultimus()
        self.ult4 = Ultimus()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.out = nn.Linear(48, 10)
    
    def forward(self, x):
      x = self.conv3(self.conv2(self.conv1(x)))
      x = self.gap(x)
      x = x.view(-1, 1*1*48)
      x = self.ult4(self.ult3(self.ult2(self.ult1(x))))
      x = self.out(x)
      return F.log_softmax(x)


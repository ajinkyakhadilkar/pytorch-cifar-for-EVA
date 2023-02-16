import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResNetBlock, self).__init__()
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    )
    self.convblock2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    )

  def forward(self, x):
    residual = x
    out = self.convblock1(x)
    out = self.convblock2(out)
    out += residual
    return out


class ResNet(nn.Module):
    expansion = 1

    def __init__(self):
        super(ResNet, self).__init__()
        # Prep Layer
        self.prep_layer = nn.Sequential( # 32x32 > 32x32 | jin=1 | RF=3
          nn.Conv2d(3, 64, 3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU()
        )
        # Layer1
        self.layer1 = nn.Sequential( 
          nn.Conv2d(64, 128, 3, padding=1), #32x32 > 32x32 | jin=1 | RF=5
          nn.MaxPool2d(2, 2), # 32x32 > 16x16 | jin=1 | RF=6 | jout=2
          nn.BatchNorm2d(128),
          nn.ReLU(),
          ResNetBlock(128, 128), # 16x16 > 16x16 | jin=2 | RF=6,146
        )
        # Layer2
        self.layer2 = nn.Sequential(
          nn.Conv2d(128, 256, 3, padding=1), # 16x16 > 16x16 | jin=2 | RF=10,18
          nn.MaxPool2d(2, 2), # 16x16 > 8x8 | jin=2 | RF=12,20 | jout=4
          nn.BatchNorm2d(256),
          nn.ReLU()
        )
        # Layer3
        self.layer3 = nn.Sequential(
          nn.Conv2d(256, 512, 3, padding=1), # 8x8 > 8x8 | jin=4 | RF=20,28
          nn.MaxPool2d(2, 2), # 8x8 > 4x4 | jin=4 | RF=24,32 | jout=8
          nn.BatchNorm2d(512),
          nn.ReLU(),
          ResNetBlock(512, 512) # 4X4>4X4 | jin=8 | RF=24,32,56,64
        )
        # Maxpool k=4
        self.mp = nn.MaxPool2d(4, 4) # 4x4 > 1x1 | jin=8 | RF=48,56,80,88
        # FC layer
        self.fc = nn.Linear(1*1*512, 10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.mp(x)
        x = x.view(-1, 1*1*512)
        out = self.fc(x)
        return out

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

from config.constant import IMAGE_SIZE

def doubleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def singleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

def fullyConnected(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        nn.ReLU(inplace=True)
    )

# VGG Model
class VGGModel(nn.Module):
    def __init__(self, classes):
        super(VGGModel, self).__init__()

        self.dblConv1 = doubleConv(3, 64)
        self.dblConv2 = doubleConv(64, 128)
        
        self.dblConv3 = doubleConv(128, 256)
        self.singleConv3 = singleConv(256, 256)

        self.dblConv4 = doubleConv(256, 512)
        self.singleConv4 = singleConv(512, 512)

        self.dblConv5 = doubleConv(512, 512)
        self.singleConv5 = singleConv(512, 512)

        self.maxPooling = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.heightWeight = IMAGE_SIZE[0] / 2**5
        self._dim = int(self.heightWeight **2 * 512)

        self.fullyConnected1 = fullyConnected(self._dim, 4096)
        self.fullyConnected2 = fullyConnected(4096, 4096)
        self.fullyConnected3 = fullyConnected(4096, 1000)

        self.out = nn.Linear(1000, classes)
    
    def forward(self, x):
        x = self.maxPooling(self.dblConv1(x))
        x = self.maxPooling(self.dblConv2(x))

        x = self.maxPooling(self.singleConv3(self.dblConv3(x)))
        x = self.maxPooling(self.singleConv4(self.dblConv4(x)))
        x = self.maxPooling(self.singleConv5(self.dblConv5(x)))

        x = self.flatten(x)

        x = self.fullyConnected3(self.fullyConnected2(self.fullyConnected1(x)))
        
        x = self.out(x)
        
        return x
    
if __name__ == "__main__":
    dummy = torch.rand((1, 3, 224, 224))
    model = VGGModel(2)
    output = model(dummy)
    print(output.shape)
    print(output)
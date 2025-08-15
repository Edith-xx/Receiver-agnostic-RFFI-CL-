import torch
import torch.nn as nn
import torch.nn.functional as F
from complex import ComplexConv
import random
import math
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv1 = ComplexConv(1, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=64)
        self.conv2 = ComplexConv(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.conv3 = ComplexConv(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(num_features=256)
        self.conv4 = ComplexConv(128, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.LazyLinear(256)
        self.fc3 = nn.LazyLinear(128)
        self.fc4 = nn.LazyLinear(6)
    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, 0.01)
        out = self.batchnorm1(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.01)
        out = self.batchnorm2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = F.leaky_relu(out, 0.01)
        out = self.batchnorm3(out)
        out = self.maxpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        features = out
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        cls = self.fc4(out)
        return features, cls
def my_resnet():
    model = DNN()
    return model






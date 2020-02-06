import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from utils.utils import logger

# CNN Model (2 conv layer) Acc 49.6%
# dee1024/pytorch-captcha-recognition
# https://github.com/dee1024/pytorch-captcha-recognition/blob/master/captcha_cnn_model.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear((200 // 8) * (70 // 8) * 64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
        )
        self.rfc = nn.Sequential(nn.Linear(1024, 2 * 10),)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out


# ResNet18 Acc 99.4%
# braveryCHR/CNN_captcha
# https://github.com/braveryCHR/CNN_captcha/blob/master/model.py
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel, track_running_stats=True),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super().__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # [3, 32, 32] -> [64, 32, 32]
        x = self.layer1(x)  # -> [64, 32, 32] -> [64, 32, 32]
        x = self.layer2(x)  # -> [128, 16, 16] -> [128, 16, 16], floor in conv2d
        x = self.layer3(x)  # -> [256, 8, 8] -> [256, 8, 8]
        x = self.layer4(x)  # -> [512, 4, 4] -> [512, 4, 4]
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(-1, 512)
        x = self.drop(x)
        y1 = self.fc1(x)  # -> [1, 10]
        y2 = self.fc2(x)
        return y1, y2

    def save(self):
        torch.save(self.state_dict(), "./model/resNet_last.pkl")

    def reload(self):
        fileList = os.listdir("./model/")
        if "resNet_last.pkl" in fileList:
            name = "./model/resNet_last.pkl"
            self.load_state_dict(torch.load(name, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))
            logger.info("ResNet Model: The latest model has been loaded.")

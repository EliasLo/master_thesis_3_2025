#import torch
import torch.nn as nn
#from torchvision.models import resnet18, vgg11
#import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes, use_batchnorm=False):
        super(LeNet, self).__init__()

        # Convolutional part:

        layers = []
        layers.append(nn.Conv2d(3, 6, kernel_size=5))
        layers.append(nn.ReLU())
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(6))
        layers.append(nn.MaxPool2d(2))

        layers.append(nn.Conv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU())
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(16))
        layers.append(nn.MaxPool2d(2))

        self.conv_layers = nn.Sequential(*layers)

        # Fully-Connected

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=False):
        super(BasicBlock, self).__init__()

        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Shortcut/Skip-Connection:
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            #
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out) if self.use_batchnorm else out
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) if self.use_batchnorm else out

        out += self.shortcut(x)  # Residual Connection
        out = F.relu(out)

        return out


class ResNet10(nn.Module):
    def __init__(self, num_classes=10, use_batchnorm=False):
        super(ResNet10, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.in_channels = 64

        # Initial Layer (conv1 + bn + relu + maxpool)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (layer1 - layer4)
        self.layer1 = self._make_layer(64, 1, stride=1)  # 1 Block
        self.layer2 = self._make_layer(128, 1, stride=2)  # 1 Block
        self.layer3 = self._make_layer(256, 1, stride=2)  # 1 Block
        self.layer4 = self._make_layer(512, 1, stride=2)  # 1 Block

        # Pooling und classification-layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        """
        BasicBlocks.
        """
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, use_batchnorm=self.use_batchnorm))
        self.in_channels = out_channels * BasicBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out) if self.use_batchnorm else out
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG11(nn.Module):
    def __init__(self, num_classes=10, use_batchnorm=False):
        super(VGG11, self).__init__()
        self.use_batchnorm = use_batchnorm

        # Feature-Extraction
        self.features = nn.Sequential(
            # Block 1
            self._make_conv_block(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            self._make_conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            self._make_conv_block(256, 512),
            self._make_conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            self._make_conv_block(512, 512),
            self._make_conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #  (Fully Connected)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels):
        """
        Erzeugt einen einzelnen Conv-Layer (3x3, stride=1, padding=1)
        ggf. mit BatchNorm und ReLU.
        """
        layers = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=1, padding=1, bias=not self.use_batchnorm)
        layers.append(conv)
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


import timm
import torch.nn as nn


def initialize_model(model_name, num_classes, device, use_batchnorm=False):
    model_name = model_name.lower()

    if model_name == "lenet":
        model = LeNet(num_classes, use_batchnorm=use_batchnorm)

    elif model_name == "vgg11":
        model = VGG11(num_classes=num_classes,
                      use_batchnorm=use_batchnorm,
                      )

    elif model_name == "resnet10":
        model = ResNet10(num_classes=num_classes,
                         use_batchnorm=use_batchnorm,
                         )

    elif model_name == "deit":
        model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)



    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)


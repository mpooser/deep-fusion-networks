import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torchvision import models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)

### user defined DFNs
class DFNetNew2(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet101(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

        self.FC = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.55),
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numCls)
        )

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.FC(x)

        return logits

# This is the prevailing model used in our experiments.
class DFNetNew1(nn.Module):
    def __init__(self, n_inputs_a = 2, n_inputs_b = 10, numCls = 17):
        super().__init__()

        resnet_a = models.resnet101(pretrained=False)
        resnet_b = models.resnet101(pretrained=False)
        resnet_fused = models.resnet101(pretrained=False)

        self.conv1_a = nn.Conv2d(n_inputs_a, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1_b = nn.Conv2d(n_inputs_b, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.encoder_a = nn.Sequential(
            self.conv1_a,
            resnet_a.bn1,
            resnet_a.relu,
            resnet_a.maxpool,
            resnet_a.layer1,
            resnet_a.layer2
        )

        self.encoder_b = nn.Sequential(
            self.conv1_b,
            resnet_b.bn1,
            resnet_b.relu,
            resnet_b.maxpool,
            resnet_b.layer1,
            resnet_b.layer2
        )

        self.post_fusion = nn.Sequential(
            resnet_fused.layer3,
            resnet_fused.layer4,
            resnet_fused.avgpool
        )

        self.FC = nn.Linear(2048, numCls)


        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x, y):
        a = self.encoder_a(x)
        b = self.encoder_b(y)
        combined = []

        for val_a, val_b in zip(a, b):
            out = avgtensors(val_a, val_b)
            #print(out)
            #print(out.shape)
            combined.append(out)

        # convert list of tensors to tensors
        combined = torch.stack((combined))
        combined = self.post_fusion(combined)
        combined = combined.view(combined.size(0), -1)
        logits = self.FC(combined)

        return logits

def avgtensors(x : torch.Tensor, y : torch.Tensor, w_x = 1, w_y = 1) -> torch.Tensor:
    return (x*w_x + y*w_y)/(w_x + w_y)

# best lr seems to be 0.001 so far but train loss slows down around 0.17
class DFNet(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_inputs, 64, kernel_size=(8, 8), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numCls, bias=True),
        )

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits

# debug
if __name__ == "__main__":
    #inputs = torch.randn((100, 12, 256, 256))

    inputs_a = torch.randn((200, 2, 256, 256))
    inputs_b = torch.randn((200, 10, 256, 256))

    # count time
    import time
    start_time = time.time()

    #net = DFNet()
    net = DFNetNew1()
    #net = DFNetNew2()
    outputs = net(inputs_a, inputs_b)

    print(f"--- {time.time() - start_time} seconds ---")
    print(f"parameter count: {count_parameters(net)}")
    #print(outputs)
    #print(outputs.shape)
    #print(inputs.shape)

    print(inputs_a.shape)
    print(inputs_b.shape)
    print(outputs.shape)
    print(50*"-")
    print(outputs)

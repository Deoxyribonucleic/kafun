import torch.nn as nn
from torchvision import models

def build_model(arch, num_classes, head="linear", freeze=False, pretrained=True):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"unknown arch: {arch}")
    in_f = m.fc.in_features
    if freeze:                              # paper technique: freeze backbone
        for p in m.parameters():
            p.requires_grad = False
    if head == "dropout":                   # paper technique: dropout head (0.5 / 0.3)
        m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    else:
        m.fc = nn.Linear(in_f, num_classes)
    return m

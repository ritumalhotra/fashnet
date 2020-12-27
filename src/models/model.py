import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet18"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet18"](pretrained=None)
        self.l0 = nn.Linear(512, 7)
        self.l1 = nn.Linear(512, 3)
        self.l2 = nn.Linear(512, 3)
        self.l3 = nn.Linear(512, 4)
        self.l4 = nn.Linear(512, 6)
        self.l5 = nn.Linear(512, 3)
        self.l6 = nn.Linear(512, 50)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        l4 = self.l4(x)
        l5 = self.l5(x)
        l6 = self.l6(x)

        return l0, l1, l2, l3, l4, l5, l6


class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        self.l0 = nn.Linear(512, 7)
        self.l1 = nn.Linear(512, 3)
        self.l2 = nn.Linear(512, 3)
        self.l3 = nn.Linear(512, 4)
        self.l4 = nn.Linear(512, 6)
        self.l5 = nn.Linear(512, 3)
        self.l6 = nn.Linear(512, 50)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        l4 = self.l4(x)
        l5 = self.l5(x)
        l6 = self.l6(x)

        return l0, l1, l2, l3, l4, l5, l6


class ResNet50(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained=None)
        self.l0 = nn.Linear(512, 7)
        self.l1 = nn.Linear(512, 3)
        self.l2 = nn.Linear(512, 3)
        self.l3 = nn.Linear(512, 4)
        self.l4 = nn.Linear(512, 6)
        self.l5 = nn.Linear(512, 3)
        self.l6 = nn.Linear(512, 50)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        l4 = self.l4(x)
        l5 = self.l5(x)
        l6 = self.l6(x)

        return l0, l1, l2, l3, l4, l5, l6

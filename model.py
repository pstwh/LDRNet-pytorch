import torch
import torch.nn as nn

import torchvision.models as models

MobileNetV2 = models.mobilenet_v2


class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, width_mult=1.4):
        super(MobileNetV2FeatureExtractor, self).__init__()

        self.model = models.mobilenet_v2(width_mult=width_mult)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x):
        return self.model(x)


class GlobalAvgPool2D(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2D, self).__init__()

    def forward(self, x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


class LDRNet(nn.Module):
    def __init__(self, points_size=100, classification_list=[1], width_mult=1.4):
        super(LDRNet, self).__init__()

        self.points_size = points_size
        self.classification_list = classification_list
        self.width_mult = width_mult

        self.backbone = MobileNetV2FeatureExtractor(width_mult=width_mult)
        if len(classification_list) > 0:
            class_size = sum(self.classification_list)
        else:
            class_size = 0
        self.global_pool = GlobalAvgPool2D()
        self.corner = nn.Linear(1792, 8)
        self.border = nn.Linear(1792, (points_size - 4) * 2)
        self.cls = nn.Linear(1792, class_size + len(self.classification_list))

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        corner_output = self.corner(x)
        border_output = self.border(x)
        cls_output = self.cls(x)
        return corner_output, border_output, cls_output


if __name__ == "__main__":
    import torch

    xx = torch.zeros((1, 3, 224, 224))
    model = LDRNet()
    y = model(xx)
    print(y)

import torch.nn as nn
from torchvision import models

def get_model(num_classes=5):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model
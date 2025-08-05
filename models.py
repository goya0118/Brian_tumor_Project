import torch
import torch.nn as nn
from torchvision import models
from config import *

class BrainMRIClassifier(nn.Module):
    """뇌 MRI 분류 모델"""
    def __init__(self, num_classes, model_name):
        super(BrainMRIClassifier, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.backbone.features[0][0] = nn.Conv2d(MODEL_CONFIG['input_channels'], 32, kernel_size=3, stride=2, padding=1, bias=False)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            
    # Forward 함수
    def forward(self, x):
        if self.model_name == 'resnet18':
            features = self.backbone.avgpool(self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))))))
            features = torch.flatten(features, 1)
            return self.classifier(features)
        elif self.model_name == 'efficientnet':
            features = self.backbone.features(x)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)
            return self.classifier(features)
        else:
            features = self.backbone(x)
            return self.classifier(features) 
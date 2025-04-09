import torch
import torch.nn as nn
import torchvision.models as models
from asam import ASAM
from auxiliary_attention import AuxiliaryAttention

class CyberSentinel(nn.Module):
    def __init__(self, num_classes=25):
        super(CyberSentinel, self).__init__()
        self.backbone = models.densenet121(pretrained=True).features
        self.asam = ASAM(1024)
        self.aux_branch = AuxiliaryAttention()
        self.classifier = nn.Sequential(
            nn.Linear(1024*5*5 + 64*10*10, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_high, x_low):
        high_feat = self.asam(self.backbone(x_high)).view(x_high.size(0), -1)
        low_feat = self.aux_branch(x_low).view(x_low.size(0), -1)
        concat_feat = torch.cat([high_feat, low_feat], dim=1)
        return self.classifier(concat_feat)

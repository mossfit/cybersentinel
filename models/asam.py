import torch
import torch.nn as nn

class ASAM(nn.Module):
    def __init__(self, channels):
        super(ASAM, self).__init__()
        self.sq_conv = nn.Conv2d(channels, channels, 9, padding=4, groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, (3,1), padding=(1,0), groups=channels)
        self.h_conv = nn.Conv2d(channels, channels, (1,3), padding=(0,1), groups=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        square = self.sq_conv(x)
        v_feature = self.v_conv(x)
        h_feature = self.h_conv(x)

        v_attn = self.sigmoid(v_feature.mean(1, keepdim=True) + v_feature.amax(1, keepdim=True))
        h_attn = self.sigmoid(h_feature.mean(1, keepdim=True) + h_feature.amax(1, keepdim=True))

        combined_attn = square * (v_attn + h_attn)
        return combined_attn

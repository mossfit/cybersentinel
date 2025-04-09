import torch
import torch.nn as nn
import torch.nn.functional as F

class ASAM(nn.Module):
    def __init__(self, channels):
        super(ASAM, self).__init__()
        self.square_conv = nn.Conv2d(channels, channels, kernel_size=9, padding=4)
        self.vert_conv = nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0))
        self.hor_conv = nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1))

    def forward(self, x):
        square = self.square_conv(x)
        vertical = self.vert_conv(x)
        horizontal = self.hor_conv(x)
        
        v_attn = torch.sigmoid(torch.cat([vertical.mean(1, keepdim=True), vertical.max(1, keepdim=True)[0]], dim=1))
        h_attn = torch.sigmoid(torch.cat([horizontal.mean(1, keepdim=True), horizontal.max(1, keepdim=True)[0]], dim=1))

        combined = square * (v_attn + h_attn)
        return combined

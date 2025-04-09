class AuxiliaryAttention(nn.Module):
    def __init__(self):
        super(AuxiliaryAttention, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.spatial_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv_layers(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention

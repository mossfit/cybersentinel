class AuxiliaryBranch(nn.Module):
    def __init__(self):
        super(AuxiliaryBranch, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.MaxPool2d(2)
        )
        self.spatial_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_block(x)
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool, _ = x.max(dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * spatial_attention

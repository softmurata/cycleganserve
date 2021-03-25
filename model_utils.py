import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ReplicationPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.InstanceNorm2d(256)
        )

    def forward(self, x):
        x = x + self.block(x)
        return x


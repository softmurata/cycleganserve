import torch.nn as nn
# from model_utils import ResidualBlock

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


class Generator(nn.Module):

    def __init__(self, img_channel=3, res_block=9):
        super(Generator, self).__init__()

        self.encode_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channel, 64, kernel_size=7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        res_blocks = [ResidualBlock() for _ in range(res_block)]

        self.res_block = nn.Sequential(
            *res_blocks
        )

        self.decode_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, img_channel, kernel_size=7, stride=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encode_block(x)
        x = self.res_block(x)
        x = self.decode_block(x)

        return x




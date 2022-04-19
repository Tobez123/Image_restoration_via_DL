import torch
from torch import nn


class DeTurbAutoEncoder(nn.Module):
    def __init__(self):
        super(DeTurbAutoEncoder, self).__init__()
        # 定义Encoder
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=32,
                      kernel_size=3, stride=1, padding=1),  # [ ,32,384,512]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1),  # [ , 64, 384, 512]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [ , 64, 192, 256]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),  # [ , 64, 192, 256]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1),  # [ , 128, 192, 256]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [ , 128, 96, 128]
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 96, 128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [ , 256, 48, 64]
            nn.BatchNorm2d(256),
        )
        # 定义Decoder
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),  # [ , 128, 48, 64]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),  # [ , 64, 96, 128]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),  # [ , 64, 96, 128]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),  # [ , 64, 192, 256]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),  # [ , 32, 192, 256]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1),  # [ , 32, 384, 512]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 3, 1, 1),  # [ , 1, 384, 512]
            nn.Sigmoid(),
        )

    # 定义网络的前向传播路径
    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder

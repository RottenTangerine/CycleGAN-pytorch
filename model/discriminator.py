from model.basic_block import BasicConv
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channel):
        super(Discriminator, self).__init__()

        layers = []
        cfd = [64, 128, 256, 512]

        for out_channel in cfd:
            layers.append(BasicConv(input_channel, out_channel, 3, 1, 1))
            input_channel = out_channel
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(512, 1, 3, padding=1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), 1)
        return x

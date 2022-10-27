import torch
import torch.nn as nn
from icecream import ic


class BasicConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, bn=True):
        super(BasicConv, self).__init__()
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_channel, output_channel, kernel_size, stride),
            nn.LeakyReLU(0.2, True)
        ]
        if bn:
            layers.append(nn.InstanceNorm2d(output_channel))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class BasicResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(BasicResidualBlock, self).__init__()

        if input_channel == output_channel:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = BasicConv(input_channel, output_channel, bn=False)

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channel, input_channel, 3),
            nn.InstanceNorm2d(input_channel),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channel, output_channel, 3),
            nn.InstanceNorm2d(output_channel),
        )

    def forward(self, x):
        x = self.shortcut(x) + self.conv(x)
        return x


class EncodeBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(EncodeBlock, self).__init__()
        self.pooling = nn.AvgPool2d((2, 2), 2)
        self.conv = BasicConv(input_channel, output_channel)
        self.residual = BasicResidualBlock(output_channel, output_channel)

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        return self.residual(x)


class DecodeBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DecodeBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = BasicConv(input_channel, output_channel)
        self.conv_past = BasicConv(input_channel, output_channel)
        self.res = BasicResidualBlock(output_channel * 2, output_channel)
        self.conv2 = BasicConv(output_channel, output_channel)

    def forward(self, x, p):
        x = self.conv(x)
        p = self.conv_past(p)
        x = self.res(torch.cat([x, p], dim=1))
        x = self.upsample(x)
        return self.conv2(x)


# input size 256x256x3
class Generator(nn.Module):
    def __init__(self, args, n_residual_block=9):
        super(Generator, self).__init__()

        # initial conv
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(args.in_channel, 64, 7),
            nn.AvgPool2d((2, 2), 2),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        # 256x256x64

        # down sampling
        self.conv1 = EncodeBlock(64, 128)
        self.conv2 = EncodeBlock(128, 256)

        # residual layers
        layers = []
        for _ in range(n_residual_block):
            layers.append(BasicResidualBlock(256, 256))
        self.residual_layers = nn.Sequential(*layers)

        # up sampling
        self.dconv2 = DecodeBlock(256, 128)
        self.dconv1 = DecodeBlock(128, 64)

        # output layer
        self.output_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, args.out_channel, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init_conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = self.residual_layers(x2)
        x = self.dconv2(x, x2)
        x = self.dconv1(x, x1)
        x = self.output_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channel):
        super(Discriminator, self).__init__()

        layers = []
        cfd = [64, 128, 256, 512]

        for out_channel in cfd:
            layers += [
                nn.Conv2d(input_channel, out_channel, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.LeakyReLU(0.2, True)
            ]
            input_channel = out_channel
        layers.append(nn.Conv2d(512, 1, 3))
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), 1)
        return x


if __name__ == '__main__':
    from config import load_config
    from dataset.dataset import GANData
    from torch.utils.data import random_split, DataLoader
    import torchvision.transforms as T
    import matplotlib.pyplot as plt

    args = load_config()
    args.dataset = '../data/apple2orange'

    dataset = GANData(args)
    train_dataset, validate_dataset = random_split(dataset,
                                                   [l := round(len(dataset) * (1 - args.test_ratio)), len(dataset) - l])
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=1, shuffle=True)
    G = Generator(args)

    img_a, img_b = (next(iter(train_loader)))
    out = G(img_a)

    trans = T.ToPILImage()
    plt.imshow(trans(img_a[0]))
    plt.show()
    plt.imshow(trans(out[0]))
    plt.show()

    D = Discriminator(3)
    print(D.eval())

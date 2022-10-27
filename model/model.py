import torch
import torch.nn as nn
from icecream import ic


class BasicConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, bn=True, activate=True):
        super(BasicConv, self).__init__()
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_channel, output_channel, kernel_size, stride)
        ]
        if bn:
            layers.append(nn.InstanceNorm2d(output_channel))
        if activate:
            layers.append(nn.LeakyReLU(0.2, True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_channel, inner_channel, input_channel=None, sub_module=None, outer_most=False,
                 inner_most=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outer_channel = outer_channel
        self.inner_channel = inner_channel
        self.outer_most = outer_most
        if not input_channel:
            input_channel = outer_channel

        down_conv = nn.Conv2d(input_channel, inner_channel, kernel_size=4, stride=2, padding=1)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = nn.InstanceNorm2d(inner_channel)

        up_relu = nn.ReLU(True)
        up_norm = nn.InstanceNorm2d(outer_channel)
        if outer_most:
            up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(inner_channel * 2, outer_channel, kernel_size=3, stride=1, padding=1)
            )
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [sub_module] + up

        elif inner_most:
            up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(inner_channel, outer_channel, kernel_size=3, stride=1, padding=1)
            )
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up
        else:
            up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(inner_channel * 2, outer_channel, kernel_size=3, stride=1, padding=1)
            )
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]
            model = down + [sub_module] + up

        self.block = nn.Sequential(*model)

    def forward(self, x):
        # ic('forward')
        if self.outer_most:
            return self.block(x)
        else:  # add skip connections
            # ic(x.shape)
            # ic(self.outer_channel, self.inner_channel)
            return torch.cat([x, self.block(x)], 1)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(512, 512, sub_module=None, inner_most=True)  # add the innermost layer
        for i in range(args.num_downs - 5):
            unet_block = UnetSkipConnectionBlock(512, 512, sub_module=unet_block)
        unet_block = UnetSkipConnectionBlock(256, 512, sub_module=unet_block)
        unet_block = UnetSkipConnectionBlock(128, 256, sub_module=unet_block)
        unet_block = UnetSkipConnectionBlock(64, 128, sub_module=unet_block)
        self.model = UnetSkipConnectionBlock(args.out_channel, 64, input_channel=args.in_channel, sub_module=unet_block,
                                             outer_most=True)  # add the outermost layer

    def forward(self, x):
        return self.model(x)


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

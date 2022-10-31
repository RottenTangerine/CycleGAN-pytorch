import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_channel, inner_channel, input_channel=None, sub_module=None, outer_most=False,
                 inner_most=False, dropout=False):
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
            if dropout:
                model = down + [sub_module] + up + [nn.Dropout(0.5)]
            else:
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
            unet_block = UnetSkipConnectionBlock(512, 512, sub_module=unet_block, dropout=True)
        unet_block = UnetSkipConnectionBlock(256, 512, sub_module=unet_block)
        unet_block = UnetSkipConnectionBlock(128, 256, sub_module=unet_block)
        unet_block = UnetSkipConnectionBlock(64, 128, sub_module=unet_block)
        self.model = UnetSkipConnectionBlock(args.out_channel, 64, input_channel=args.in_channel, sub_module=unet_block,
                                             outer_most=True)  # add the outermost layer

    def forward(self, x):
        return self.model(x)

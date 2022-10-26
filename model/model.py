import torch.nn as nn


class BasicResidualBlock(nn.Module):
    def __init__(self, input_channel):
        super(BasicResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channel, input_channel, 3),
            nn.InstanceNorm2d(input_channel),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channel, input_channel, 3),
            nn.InstanceNorm2d(input_channel),
        )

    def forward(self, x):
        return x + self.conv(x)


# input size 256x256x3
class Generator(nn.Module):
    def __init__(self, args, n_residual_block=9):
        super(Generator, self).__init__()

        # initial conv
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(args.in_channel, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        # 256x256x64

        # down sampling
        _in_channel = 64
        cfd = [64, 128, 128, 256]
        layers = []
        for _out_channel in cfd:
            layers += [
                nn.Conv2d(_in_channel, _out_channel, 3, 2, 1),
                nn.InstanceNorm2d(_out_channel),
                nn.ReLU(True)
            ]
            _in_channel = _out_channel
        self.down_sampling = nn.Sequential(*layers)

        # residual layers
        layers = []
        for _ in range(n_residual_block):
            layers.append(BasicResidualBlock(_in_channel))
        self.residual_layers = nn.Sequential(*layers)

        # up sampling
        _in_channel = 256
        layers = []
        for _out_channel in cfd[::-1]:
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(_in_channel, _out_channel, 3, 1, 1),
                nn.InstanceNorm2d(_out_channel)
            ]
            _in_channel = _out_channel
        self.up_sampling = nn.Sequential(*layers)

        # output layer
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, args.out_channel, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.down_sampling(x)
        x = self.residual_layers(x)
        x = self.up_sampling(x)
        x = self.output_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channel):
        super(Discriminator, self).__init__()

        layers = []
        cfd = [64, 128, 256, 512]

        for out_channel in cfd:
            layers += [
                nn.Conv2d(input_channel, out_channel, 3),
                nn.InstanceNorm2d(out_channel),
                nn.LeakyReLU(0.2, True)
            ]
            input_channel = out_channel
        layers.append(nn.Conv2d(512, 1, 3))
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    from config import load_config
    from dataset.dataset import GANData
    from torch.utils.data import random_split, DataLoader
    import torchvision.transforms as T
    import matplotlib.pyplot as plt

    args = load_config()
    args.dataset = '../data/apple2orange'
    print(args)

    dataset = GANData(args)
    train_dataset, validate_dataset = random_split(dataset,
                                                   [l := round(len(dataset) * (1 - args.test_ratio)), len(dataset) - l])
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=8, shuffle=True)
    G = Generator(args)

    img_a, img_b = (next(iter(train_loader)))
    print(img_a.shape)
    out = G(img_a)
    print(out.shape)

    trans = T.ToPILImage()
    plt.imshow(trans(img_a[0]))
    plt.show()
    plt.imshow(trans(out[0]))
    plt.show()

    D = Discriminator(3)
    print(D.eval())

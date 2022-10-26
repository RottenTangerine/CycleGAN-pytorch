import argparse


def load_config():
    args = argparse.ArgumentParser()

    args.add_argument('--dataset', type=str, default='data/apple2orange')
    args.add_argument('--test_ratio', type=float, default=0.2)

    # device
    args.add_argument('--cuda', type=bool, default=True, help='use cuda training')

    # models
    args.add_argument('--in_channel', type=int, default=3, help='number of input channel')
    args.add_argument('--out_channel', type=int, default=3, help='number of output channel')

    # train
    args.add_argument('--epochs', type=int, default='80', help='number of training epochs')
    args.add_argument('--batch_size', type=int, default='2', help='number of batch size')
    args.add_argument('--lr_g', type=float, default='2e-4', help='learning rate of generator')
    args.add_argument('--lr_d', type=float, default='5e-6', help='learning rate of discriminator')
    args.add_argument('--model', type=str, default='basic', help='choose the model used to be train')

    args.add_argument('--print_batch', type=int, default=50, help='print interception')

    args.add_argument('--loss_weight_identity', type=float, default=5.0, help='weight of the Identity loss')
    args.add_argument('--loss_weight_gan', type=float, default=10.0, help='weight of the GAN loss')
    args.add_argument('--loss_weight_cycle', type=float, default=10.0, help='weight of the cycle loss')

    return args.parse_args(args=[])



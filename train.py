import itertools
import random
import time
import os

import torch
from config import load_config
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataset.dataset import GANData

from model import discriminator, generator
from model.utils import init_net

from torchvision import transforms
from utils.image_pool import ImagePool

# from icecream import ic

args = load_config()
train_id = int(time.time())
resume_epoch = 0
print(f'Training ID: {train_id}')

train_dataset = GANData(args)
test_dataset = GANData(args, train=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

print(f'Train data batches: {len(train_loader)}')

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

# model
G_A2B = init_net(generator.Generator(args), args.init_type, args.init_gain).to(device)
G_B2A = init_net(generator.Generator(args), args.init_type, args.init_gain).to(device)
D_A = init_net(discriminator.Discriminator(args.in_channel), args.init_type, args.init_gain).to(device)
D_B = init_net(discriminator.Discriminator(args.out_channel), args.init_type, args.init_gain).to(device)

# retrained / continuous training
try:
    most_recent_check_point = os.listdir('checkpoint')[-1]
    ckpt_path = os.path.join('checkpoint', most_recent_check_point)
    check_point = torch.load(ckpt_path)
    # load model
    G_A2B.load_state_dict(check_point['G_A2B_state_dict'])
    G_B2A.load_state_dict(check_point['G_B2A_state_dict'])
    D_A.load_state_dict(check_point['D_A_state_dict'])
    D_B.load_state_dict(check_point['D_B_state_dict'])
    resume_epoch = check_point['epoch']
    print(f'Successfully load checkpoint {most_recent_check_point}, '
          f'start training from epoch {resume_epoch + 1}')
except:
    print('fail to load checkpoint, train from zero beginning')

# optimizer and schedular
optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                               lr=args.lr_g, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

lr_scheduler_G = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0=10, T_mult=2, eta_min=1e-5)
lr_scheduler_D_A = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D_A, T_0=10, T_mult=2, eta_min=1e-5)
lr_scheduler_D_B = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D_B, T_0=10, T_mult=2, eta_min=1e-5)
# lr_scheduler_D_A = lr_scheduler.ExponentialLR(optimizer_D_A, gamma=0.9)
# lr_scheduler_D_B = lr_scheduler.ExponentialLR(optimizer_D_B, gamma=0.9)

for _ in range(resume_epoch):
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

# criterion
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.Tensor
target_real = Variable(Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

# image pool to enhance the robustness of training
fake_A_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images
fake_B_pool = ImagePool(args.pool_size)

print('***start training***')
# training
for epoch in range(resume_epoch + 1, args.epochs):
    epoch_start_time = time.time()
    print(f'{"*" * 20} Start epoch {epoch}/{args.epochs} {"*" * 20}')

    for i, (real_a, real_b) in enumerate(train_loader):
        real_a = real_a.to(device)
        real_b = real_b.to(device)

        # a -> b -> a
        fake_b = G_A2B(real_a)
        fake_b = fake_B_pool.query(fake_b)
        rec_a = G_B2A(fake_b)

        # b -> a -> b
        fake_a = G_B2A(real_b)
        fake_a = fake_A_pool.query(fake_a)
        rec_b = G_A2B(fake_a)

        #  Generator
        optimizer_G.zero_grad()
        #  #  Identity
        same_B = G_A2B(real_b)
        loss_identity_B = criterion_identity(same_B, real_b) * args.loss_weight_GB
        same_A = G_B2A(real_a)
        loss_identity_A = criterion_identity(same_A, real_a) * args.loss_weight_GA

        #  #  GAN loss
        predict_fake = D_B(fake_b)
        loss_GAN_A2B = criterion_GAN(predict_fake, target_real)

        predict_fake = D_A(fake_a)
        loss_GAN_B2A = criterion_GAN(predict_fake, target_real)

        #  # Cycle loss
        loss_cycle_ABA = criterion_cycle(rec_a, real_a) * args.loss_weight_GA
        loss_cycle_BAB = criterion_cycle(rec_b, real_b) * args.loss_weight_GB

        loss = args.loss_weight_gan * (loss_GAN_A2B + loss_GAN_B2A) \
            + args.loss_weight_identity * (loss_identity_A + loss_identity_B) \
            + args.loss_weight_cycle * (loss_cycle_ABA + loss_cycle_BAB)

        loss.backward()
        optimizer_G.step()

        # Discriminator B
        optimizer_D_B.zero_grad()
        pred_real = D_B(real_b)
        loss_pred_real = criterion_GAN(pred_real, target_real)
        pred_fake = D_B(fake_b.detach())  # detach there to avoid computed the graph twice
        loss_pred_fake = criterion_GAN(pred_fake, target_fake)

        loss_D_B = (loss_pred_real + loss_pred_fake) * 0.5

        loss_D_B.backward()
        optimizer_D_B.step()

        # Discriminator A
        optimizer_D_A.zero_grad()
        pred_real = D_A(real_a)
        loss_pred_real = criterion_GAN(pred_real, target_real)
        pred_fake = D_A(fake_a.detach())
        loss_pred_fake = criterion_GAN(pred_fake, target_fake)
        # ic(pred_real, pred_fake, target_real, target_fake)

        loss_D_A = (loss_pred_real + loss_pred_fake) * 0.5

        loss_D_A.backward()
        optimizer_D_A.step()

        if i % args.print_interval == 0:
            print(f'epoch: {epoch}/{args.epochs}\tbatch: {i}/{len(train_loader)}\t'
                  f'loss_G: {loss:0.6f}\tloss_D_A: {loss_D_A:0.6f}\tloss_D_B: {loss_D_B:0.6f}\t'
                  f'|| learning rate_G: {optimizer_G.state_dict()["param_groups"][0]["lr"]:0.6f}\t'
                  f'learning rate_D_A: {optimizer_D_A.state_dict()["param_groups"][0]["lr"]:0.8f}\t'
                  f'learning rate_D_B: {optimizer_D_B.state_dict()["param_groups"][0]["lr"]:0.8f}\t')

    with torch.no_grad():
        real_a, real_b = random.choice(test_dataset)

        real_a = real_a.to(device).unsqueeze(0)
        real_b = real_b.to(device).unsqueeze(0)

        # a -> b -> a
        fake_b = G_A2B(real_a)
        rec_a = G_B2A(fake_b)

        # b -> a -> b
        fake_a = G_B2A(real_b)
        rec_b = G_A2B(fake_a)

        #  Generator
        same_B = G_A2B(real_b)
        same_A = G_B2A(real_a)

        img_a = torch.cat([real_a, fake_b, rec_a, same_A], dim=3)
        img_b = torch.cat([real_b, fake_a, rec_b, same_B], dim=3)
        img = torch.cat([img_a, img_b], dim=2)

        os.makedirs('output', exist_ok=True)
        trans = transforms.ToPILImage()
        trans(img[0]).save(f'output/{train_id}_{epoch}.jpg')



    # scheduler
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    print(f'End of epoch: {epoch}/{args.epochs}\t time taken: {time.time() - epoch_start_time:0.2f}')

    # save ckpt
    os.makedirs('checkpoint', exist_ok=True)
    torch.save({'epoch': epoch,
                'G_A2B_state_dict': G_A2B.state_dict(),
                'G_B2A_state_dict': G_B2A.state_dict(),
                'D_A_state_dict': D_A.state_dict(),
                'D_B_state_dict': D_B.state_dict(),
                }, f'checkpoint/{train_id}_{epoch:03d}.pt')

# save model
os.makedirs('trained_model', exist_ok=True)
torch.save(G_A2B.state_dict(), f'./trained_model/G_A2B_{train_id}.pth')
torch.save(G_B2A.state_dict(), f'./trained_model/G_B2A_{train_id}.pth')

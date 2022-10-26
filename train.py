import itertools
import time
import os

import torch
from config import load_config
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataset.dataset import GANData
from model.model import Generator, Discriminator

from torchvision import transforms
from PIL import Image

args = load_config()
train_id = int(time.time())
resume_epoch = 0
print(f'Training ID: {train_id}')

dataset = GANData(args)
train_dataset, validate_dataset = random_split(dataset,
                                               [l := round(len(dataset) * (1 - args.test_ratio)),
                                                len(dataset) - l])
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=args.batch_size, shuffle=True)

print(f'Train data batches: {len(train_loader)}, Validate Data batches:{len(validate_loader)}')

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

# model
G_A2B = Generator(args).to(device)
G_B2A = Generator(args).to(device)
D_A = Discriminator(args.in_channel).to(device)
D_B = Discriminator(args.out_channel).to(device)

# retrained / continuous training
try:
    most_recent_check_point = os.listdir('checkpoint')[-1]
    ckpt_path = os.path.join('checkpoint', most_recent_check_point)
    check_point = torch.load(ckpt_path)
    # load model
    G_A2B.load_state_dict(check_point['G_A2B_state_dict'])
    G_B2A.load_state_dict(check_point['G_B2A_state_dict'])
    # D_A.load_state_dict(check_point['D_A_state_dict'])
    # D_B.load_state_dict(check_point['D_B_state_dict'])
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

lr_scheduler_G = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0=10, T_mult=2, eta_min=5e-4)
# lr_scheduler_D_A = lr_scheduler.ExponentialLR(optimizer_D_A, gamma=0.9)
# lr_scheduler_D_B = lr_scheduler.ExponentialLR(optimizer_D_B, gamma=0.9)

# criterion
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.Tensor
target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)

print('***start training***')
# training
for epoch in range(resume_epoch + 1, args.epochs):
    for i, (img_a, img_b) in enumerate(train_loader):
        img_a = img_a.to(device)
        img_b = img_b.to(device)

        # Generator
        # Identity
        same_B = G_A2B(img_b)
        loss_identity_B = criterion_identity(same_B, img_b)
        same_A = G_B2A(img_a)
        loss_identity_A = criterion_identity(same_A, img_a)

        # GAN loss
        fake_B = G_A2B(img_a)
        predict_fake = D_B(fake_B)
        loss_GAN_A2B = criterion_GAN(predict_fake, target_real)

        fake_A = G_B2A(img_b)
        predict_fake = D_A(fake_A)
        loss_GAN_B2A = criterion_GAN(predict_fake, target_real)

        # Cycle loss
        recovered_A = G_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, img_a)

        recovered_B = G_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, img_b)

        loss = args.loss_weight_identity * (loss_identity_A + loss_identity_B) \
            + args.loss_weight_gan * (loss_GAN_A2B + loss_GAN_B2A) \
            + args.loss_weight_cycle * (loss_cycle_ABA + loss_cycle_BAB)

        loss.backward()
        optimizer_G.step()

        # Discriminator B
        optimizer_D_B.zero_grad()
        pred_real = D_B(img_b)
        loss_pred_real = criterion_GAN(pred_real, target_real)
        pred_fake = D_B(fake_B.detach())  # detach there to avoid computed the graph twice
        loss_pred_fake = criterion_GAN(pred_fake, target_fake)

        loss_D_B = (loss_pred_real + loss_pred_fake) * 0.5

        loss_D_B.backward()
        optimizer_D_B.step()

        # Discriminator A
        optimizer_D_A.zero_grad()
        pred_real = D_A(img_a)
        loss_pred_real = criterion_GAN(pred_real, target_real)
        pred_fake = D_A(fake_A.detach())
        loss_pred_fake = criterion_GAN(pred_fake, target_fake)

        loss_D_A = (loss_pred_real + loss_pred_fake) * 0.5

        loss_D_A.backward()
        optimizer_D_A.step()

        if (i) % args.print_batch == 0:
            print(f'epoch: {epoch}/{args.epochs}\tbatch: {i}/{len(train_loader)}\t'
                  f'loss_G: {loss:0.6f}\tloss_D_A: {loss_D_A:0.6f}\tloss_D_B: {loss_D_B:0.6f}\t'
                  f'|| learning rate_G: {optimizer_G.state_dict()["param_groups"][0]["lr"]:0.6f}\t'
                  f'learning rate_D_A: {optimizer_D_A.state_dict()["param_groups"][0]["lr"]:0.6f}\t'
                  f'learning rate_D_B: {optimizer_D_B.state_dict()["param_groups"][0]["lr"]:0.6f}\t')

            os.makedirs('output', exist_ok=True)
            trans = transforms.ToPILImage()
            trans(img_a[0]).save(f'output/{train_id}_{epoch}_{i}_original.jpg')
            trans(fake_B[0]).save(f'output/{train_id}_{epoch}_{i}_fake.jpg')



    # scheduler
    # lr_scheduler_G.step()
    # lr_scheduler_D_A.step()
    # lr_scheduler_D_B.step()

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

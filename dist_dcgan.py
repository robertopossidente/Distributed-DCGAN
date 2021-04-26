import torch
import os
import random
import numpy as np
import argparse
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nz, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def get_dataset(dataset_name: str, dataroot: str, image_size: int):
    if dataset_name in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(
            root=dataroot, transform=transforms.Compose([
                transforms.Resize(image_size), transforms.CenterCrop(image_size),
                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), \
                    (0.5, 0.5, 0.5)),]))
        nc = 3
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(
            root=dataroot, download=False, transform=transforms.Compose([
                transforms.Resize(image_size), transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
        nc=3
    elif dataset_name == 'mnist':
        dataset = dset.MNIST(
        root=dataroot, download=False, transform=transforms.Compose([
                transforms.Resize(image_size), transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),]))
        nc=1
    else:
        raise ValueError(f"Invalid dataset {dataset_name}")

    print(f"Dataset {dataset_name} loaded")
    return dataset, nc

def set_random_seeds(random_seed: int = 0, use_cuda: bool = False):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Using seed: {random_seed}")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def main():
    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'lsun', 'mnist', \
                        'imagenet', 'folder', 'lfw', 'fake'])
    parser.add_argument('--dataroot', required=True, help='Path to dataset')
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the \
                        torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", \
                        default=25)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",\
                        default=32)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=0.0002)
    parser.add_argument('--image_size', type=int, default=64, help='The height / width of the \
                        input image to network')
    parser.add_argument('--seed', type=int, default=0, help='Set a manual random seed')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for adam.')
    parser.add_argument('--nz', type=int, default=100, help='Size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--out_folder', default='.', help='folder to output images')
    parser.add_argument('--max_workers', default=4, type=int, help='Number of workers to perform loading')
    argv = parser.parse_args()
    print(argv)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(argv.seed, argv.cuda)
    print(f"Using GPU: {argv.cuda}")

    # Initializes the distributed backend which will take care of sychronizing nodes
    torch.distributed.init_process_group(backend="gloo")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"Rank: {rank}. World Size: {world_size}")

    if rank == 0:
        os.makedirs(argv.out_folder, exist_ok=True)

    # Load datasets and wrap into Distributed Sampler
    train_dataset, n_classes = get_dataset(argv.dataset, argv.dataroot, argv.image_size)
    train_sampler = DistributedSampler(dataset=train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=argv.batch_size, sampler=train_sampler, \
                                num_workers=argv.max_workers)

    # Create models
    device = torch.device(f"cuda:{argv.local_rank}" if argv.cuda else "cpu")
    netG = Generator(argv.nz, argv.ngf, n_classes).to(device)
    netD = Discriminator(argv.nz, argv.ndf, n_classes).to(device)
    # Wrap as DistributedDataParallel
    netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[argv.local_rank] if argv.cuda else None, \
                            output_device=argv.local_rank if argv.cuda else None)
    netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[argv.local_rank] if argv.cuda else None, \
                            output_device=argv.local_rank if argv.cuda else None)
    # Set train mode to networks
    netG.train()
    netD.train()

    # Create loss and optimizer
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=argv.learning_rate, betas=(argv.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=argv.learning_rate, betas=(argv.beta1, 0.999))

    # Aditional values
    fixed_noise = torch.randn(argv.batch_size, argv.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # Lets start all together. Optimizers all have barrier also
    torch.distributed.barrier()

    for epoch in range(argv.num_epochs):
        epoch_start_time = time.time()
        print(f"Rank: {rank}, Epoch: {epoch}, Training ...")
        for i, data in enumerate(train_loader):
            iteration_start_time = time.time()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, argv.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            # A barrier here
            iteration_end_time = time.time()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            # A barrier here
            optimizerG.step()

            iteration_end_time = time.time()-iteration_start_time
            print(f"[epoch: {epoch}/{argv.num_epochs}][iteration: {i}/{len(train_loader)}][rank: {rank}] " \
                  f"Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}, " \
                  f"D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}, " \
                  f"iteration time: {iteration_end_time:.4f}s")

            if i%100 == 0:
                vutils.save_image(real_cpu, f'{argv.out_folder}/real_samples_rank_{rank}_epoch_{epoch}_iter_{i}.png', normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), f'{argv.out_folder}/fake_samples_rank_{rank}_epoch_{epoch}_iter_{i}.png', normalize=True)
                torch.distributed.barrier()

        epoch_end_time = time.time()-epoch_start_time
        print(f"[rank: {rank}] Epoch {epoch} took: {epoch_end_time:.4f} seconds")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
